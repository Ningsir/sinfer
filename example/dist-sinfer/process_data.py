import argparse
import time
import os
import sys
from typing import List
import torch as th
import torch.distributed as dist
import numpy as np
import json
from torch_sparse import SparseTensor
from ogb.nodeproppred import PygNodePropPredDataset
import torch.multiprocessing as mp

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
from sinfer.distributed import alltoallv
from sinfer.cpp_core import rand_assign_partition_nodes


def load_partitioned_edges(files: List[str]):
    all_edges = []
    for file in files:
        edges = np.fromfile(file, dtype=np.int64).reshape(-1, 2)
        all_edges.append(edges)

    return np.concatenate(all_edges)


def get_partition_indices(partition_num, random=False) -> List[int]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        if random:
            objects = th.randperm(partition_num, dtype=th.int64)
        else:
            objects = th.arange(partition_num, dtype=th.int64)
        splits = th.tensor([partition_num // world_size] * world_size, dtype=th.int64)
        if partition_num % world_size != 0:
            indices = th.randperm(world_size)
            splits[indices[: partition_num % world_size]] += 1
        objects = objects.split(splits.tolist())
    else:
        objects = [None for _ in range(world_size)]
    dist.broadcast_object_list(objects)
    return objects[rank].tolist()


def create_degree(nodes, local_degree):
    world_size = dist.get_world_size()
    dst = nodes % world_size
    send_nodes = []
    send_degrees = []
    dst_indices = [th.nonzero(dst == i, as_tuple=True)[0] for i in range(world_size)]
    for i in range(world_size):
        send_nodes.append(nodes[dst_indices[i]])
        send_degrees.append(local_degree[dst_indices[i]])
    # run alltoallv
    recv_nodes = alltoallv(send_nodes)
    recv_degrees = alltoallv(send_degrees)
    # update global degree
    indices = [
        th.div(_nodes, world_size, rounding_mode="floor") for _nodes in recv_nodes
    ]
    max_nodes = th.cat(recv_nodes).max().item()
    global_degree = th.zeros([(max_nodes + world_size) // world_size], dtype=th.int64)
    for i in range(world_size):
        global_degree[indices[i]] += recv_degrees[i]
    # send the global degree to the rank its local degree from
    send_global_degree = [global_degree[indices[i]] for i in range(world_size)]
    recv_global_degree = alltoallv(send_global_degree)
    global_degree = th.zeros_like(local_degree)
    for i in range(world_size):
        global_degree[dst_indices[i]] += recv_global_degree[i]
    return local_degree, global_degree


def mapping(coo_row, coo_col, all_nodes, feat, labels, train_idx, val_idx, test_idx):
    map_id = np.arange(all_nodes.shape[0], dtype=np.int64)
    total_num_nodes = all_nodes.max().item()
    total_num_nodes = (
        max(
            total_num_nodes,
            train_idx.max().item(),
            val_idx.max().item(),
            test_idx.max().item(),
        )
        + 1
    )
    global_to_local_map = -np.ones(total_num_nodes, dtype=np.int64)
    global_to_local_map[all_nodes] = map_id
    map_feat = feat[all_nodes]
    map_labels = labels[all_nodes]

    # global to local
    map_coo_row = global_to_local_map[coo_row]
    map_coo_col = global_to_local_map[coo_col]
    map_train_idx = global_to_local_map[train_idx]
    map_val_idx = global_to_local_map[val_idx]
    map_test_idx = global_to_local_map[test_idx]
    print(map_coo_row)
    print(map_coo_col)
    return (
        map_coo_row,
        map_coo_col,
        map_feat,
        map_labels,
        map_train_idx,
        map_val_idx,
        map_test_idx,
        all_nodes.shape[0],
    )


def idx_split(map_idx):
    """
    随机划分训练集、验证集或者测试集: 如果一个节点在多个分区中, 则随机划分到其中一个分区中
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    nodes_to_parts = th.zeros(map_idx.shape[0], world_size, dtype=th.int64)
    # map_idx等于-1表示不在该分区上
    indices = th.nonzero(
        map_idx != -th.ones_like(map_idx, dtype=th.int64), as_tuple=True
    )[0]
    # nodes_to_parts[i][j] == 1表示节点i在分区j内
    nodes_to_parts[indices, dist.get_rank()] = 1
    if rank == 0:
        gathered_data = [th.zeros_like(nodes_to_parts) for _ in range(world_size)]
    else:
        gathered_data = None
    dist.gather(nodes_to_parts, gathered_data, dst=0)
    unique_nodes = []
    scatter_size = []
    if dist.get_rank() == 0:
        # 更新nodes_to_parts
        for i in range(world_size - 1):
            nodes_to_parts = th.logical_or(gathered_data[i], nodes_to_parts).to(
                th.int64
            )
        nodes_to_part_id_map = rand_assign_partition_nodes(nodes_to_parts)
        for id in range(world_size):
            unique_nodes.append(
                th.nonzero(nodes_to_part_id_map == id, as_tuple=True)[0]
            )
            scatter_size.append(unique_nodes[-1].shape)
    output_size = [None]
    dist.scatter_object_list(output_size, scatter_size, src=0)
    if rank == 0:
        output_tensor = unique_nodes[0]
        for i in range(1, world_size):
            dist.send(unique_nodes[i], dst=i)
    else:
        output_tensor = th.zeros(output_size[0], dtype=th.int64)
        dist.recv(output_tensor, src=0)
    nodes = map_idx[output_tensor]
    return nodes


def main(rank, world_size, dataset, part_path, num_parts, out_path):
    out_path = os.path.join(out_path, f"part-{rank}")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # TODO: 处理孤立点
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    feat = dataset[0].x.numpy()
    labels = dataset[0].y.numpy()
    num_classes = dataset.num_classes
    origin_total_num_nodes = dataset.num_nodes
    partition_indices = get_partition_indices(num_parts, random=True)
    filenames = [
        os.path.join(part_path, f"{idx}.partitioned_edges") for idx in partition_indices
    ]
    edges = load_partitioned_edges(filenames)
    edges = th.from_numpy(edges)
    coo_row = edges[:, 0]
    coo_col = edges[:, 1]
    # get local degree
    # TODO: 支持有向图, 该方法求的度是in_degree+out_degree
    nodes, local_degree = th.unique(edges.flatten(), return_counts=True)
    local_degree, global_degree = create_degree(nodes, local_degree)
    local_degree = local_degree.numpy()
    global_degree = global_degree.numpy()
    (
        map_coo_row,
        map_coo_col,
        map_feat,
        map_labels,
        map_train_idx,
        map_val_idx,
        map_test_idx,
        num_nodes,
    ) = mapping(coo_row, coo_col, nodes, feat, labels, train_idx, val_idx, test_idx)
    # 添加反向边
    map_coo_row_ = np.concatenate((map_coo_row, map_coo_col))
    map_coo_col_ = np.concatenate((map_coo_col, map_coo_row))
    map_coo_row = map_coo_row_
    map_coo_col = map_coo_col_
    # 划分训练集、测试集、验证集
    split_train_idx = idx_split(th.from_numpy(map_train_idx)).numpy()
    split_test_idx = idx_split(th.from_numpy(map_test_idx)).numpy()
    split_val_idx = idx_split(th.from_numpy(map_val_idx)).numpy()
    print(
        f"rank: {rank}, split train idx: {split_train_idx.size}, split val idx: {split_val_idx.size}, split test idx: {split_test_idx.size}"
    )
    # coo转为csc
    sparse_tensor = SparseTensor.from_edge_index(
        (th.from_numpy(map_coo_row), th.from_numpy(map_coo_col)),
        sparse_sizes=(num_nodes, num_nodes),
    )
    indptr, indices, _ = sparse_tensor.csc()
    assert indptr.shape[0] == num_nodes + 1, f"{indptr.shape[0]} != {num_nodes + 1}"
    indptr = indptr.numpy()
    indices = indices = indices.numpy()

    map_coo = np.concatenate((map_coo_row, map_coo_col))
    map_coo = map_coo.reshape(2, -1)
    # 写入文件
    indptr.tofile(os.path.join(out_path, "indptr.bin"))
    indices.tofile(os.path.join(out_path, "indices.bin"))
    map_coo.tofile(os.path.join(out_path, "coo.bin"))
    map_feat.tofile(os.path.join(out_path, "feat.bin"))
    map_labels.tofile(os.path.join(out_path, "labels.bin"))
    split_train_idx.tofile(os.path.join(out_path, "train_idx.bin"))
    split_val_idx.tofile(os.path.join(out_path, "val_idx.bin"))
    split_test_idx.tofile(os.path.join(out_path, "test_idx.bin"))
    local_degree.tofile(os.path.join(out_path, "local_degree.bin"))
    global_degree.tofile(os.path.join(out_path, "global_degree.bin"))
    origin_nodes = nodes.numpy()
    origin_nodes.tofile(os.path.join(out_path, "origin_nodes.bin"))
    config = {
        "num_nodes": num_nodes,
        "origin_total_num_nodes": origin_total_num_nodes,
        "feat_dim": map_feat.shape[-1],
        "num_classes": num_classes
        if num_classes is not None
        else np.unique(labels).shape[0],
        "indptr_dtype": str(indptr.dtype),
        "indices_dtype": str(indices.dtype),
        "coo_dtype": str(map_coo.dtype),
        "feat_dtype": str(map_feat.dtype),
        "labels_dtype": str(map_labels.dtype),
        "train_idx_dtype": str(map_train_idx.dtype),
        "val_idx_dtype": str(map_val_idx.dtype),
        "test_idx_dtype": str(map_test_idx.dtype),
        "origin_nodes_dtype": str(origin_nodes.dtype),
        "local_degree_dtype": str(local_degree.dtype),
        "global_degree_dtype": str(global_degree.dtype),
    }
    with open(os.path.join(out_path, "conf.json"), "w") as f:
        json.dump(config, f)
    print(f"Done! The dataset is saved in {out_path}")


if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="dataset name: ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument("--data-path", type=str, default="/mnt/ningxin/data")
    argparser.add_argument("--num-parts", type=int, default=8)
    argparser.add_argument("--output-num-parts", type=int, default=2)
    argparser.add_argument(
        "--part-path",
        type=str,
        default="/mnt/ningxin/data/ssd/ogbn-products-dne-part8/",
        help="path of dne partitions",
    )
    argparser.add_argument("--output-path", type=str, default="/mnt/ningxin/data/dist/")
    args = argparser.parse_args()
    dataset = PygNodePropPredDataset(args.dataset, args.data_path)
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    print(
        f"train size: {train_idx.size(0)}, valid size: {val_idx.size(0)}, test size: {test_idx.size(0)}"
    )
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = args.output_num_parts
    out_path = os.path.join(args.output_path, args.dataset)
    mp.spawn(
        main,
        args=(world_size, dataset, args.part_path, args.num_parts, out_path),
        nprocs=world_size,
        join=True,
    )
