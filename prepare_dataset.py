import pymetis
import time
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
import scipy
import numpy as np
import json
import torch
import os
from torch_sparse import SparseTensor

from sinfer.utils import coo_to_adj_list
from sinfer.cpp_core import rand_assign_partition_nodes


def load_dne_part_nodes(dne_part_path, num_nodes, num_parts):
    nodes = []
    for id in range(num_parts):
        filename = os.path.join(dne_part_path, f"{id}.partitioned_nodes")
        node = np.unique(np.fromfile(filename, dtype=np.int64))
        nodes.append(node)
    nodes_to_parts = np.zeros((num_nodes, num_parts), dtype=np.int64)
    for id, node in enumerate(nodes):
        nodes_to_parts[node, id] = 1
    # nodes_to_part_id_map[i] 表示顶点i被划分到的分区ID
    nodes_to_part_id_map = rand_assign_partition_nodes(torch.from_numpy(nodes_to_parts))
    nodes_to_part_id_map = nodes_to_part_id_map.numpy()
    unique_nodes = []
    offsets = [0]
    count = 0
    for id in range(num_parts):
        mask = nodes_to_part_id_map == id
        unique_nodes.append(np.nonzero(mask)[0])
        count += unique_nodes[-1].shape[0]
        offsets.append(count)
    for i in range(num_parts):
        x = np.intersect1d(nodes[i], unique_nodes[i])
        print(x.shape)
    return unique_nodes, offsets


def process_products_with_dne(args, dne_part_path):
    num_parts = args.num_parts
    out_path = os.path.join(
        args.data_path, args.dataset + "-ssd-infer-dne-part{}".format(num_parts)
    )
    data = DglNodePropPredDataset(name=args.dataset, root=args.data_path)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]
    print(graph)

    nfeat = graph.ndata.pop("feat").numpy()
    labels = labels[:, 0].numpy()
    src = graph.edges()[0].numpy()
    dst = graph.edges()[1].numpy()
    num_parts = args.num_parts
    # 加载dne的分区节点
    nodes_part, offsets = load_dne_part_nodes(
        dne_part_path, graph.num_nodes(), num_parts
    )
    np_offsets = np.array(offsets, dtype=np.int64)
    __process_data(
        src,
        dst,
        nfeat,
        labels,
        out_path,
        train_idx,
        val_idx,
        test_idx,
        nodes_part,
        np_offsets,
    )
    print("process dne data done")


def process_dgl_data(args):
    num_parts = args.num_parts
    out_path = os.path.join(
        args.data_path, args.dataset + "-ssd-infer-part{}".format(num_parts)
    )
    data = DglNodePropPredDataset(name=args.dataset, root=args.data_path)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]
    print(graph)

    nfeat = graph.ndata.pop("feat").numpy()
    labels = labels[:, 0].numpy()
    src = graph.edges()[0].numpy()
    dst = graph.edges()[1].numpy()
    indptr, indices, edges = graph.adj_sparse("csr")
    indptr = indptr.tolist()
    indices = indices.tolist()
    # adj = coo_to_adj_list(src, dst)
    start = time.time()
    n_cuts, membership = pymetis.part_graph(num_parts, xadj=indptr, adjncy=indices)
    print("#n_cuts: {}, partition time: {:.4f} s".format(n_cuts, time.time() - start))
    nodes_part = []
    offsets = [0]
    total = 0
    for i in range(num_parts):
        part = np.argwhere(np.array(membership) == i).ravel()
        nodes_part.append(part)
        total += part.shape[0]
        offsets.append(total)
    np_offsets = np.array(offsets, dtype=np.int64)
    print(np_offsets)
    __process_data(
        src,
        dst,
        nfeat,
        labels,
        out_path,
        train_idx,
        val_idx,
        test_idx,
        nodes_part,
        np_offsets,
        num_classes=data.num_classes,
    )


def __process_data(
    coo_row,
    coo_col,
    feat,
    labels,
    out_path,
    train_idx,
    val_idx,
    test_idx,
    nodes_part,
    np_offsets,
    num_classes=None,
):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    # 按照nodes_part中的顺序对顶点ID进行重映射
    (
        map_coo_row,
        map_coo_col,
        map_feat,
        map_labels,
        map_train_idx,
        map_val_idx,
        map_test_idx,
        num_nodes,
    ) = mapping(
        coo_row, coo_col, nodes_part, feat, labels, train_idx, val_idx, test_idx
    )
    map_coo = np.concatenate((map_coo_row, map_coo_col))
    map_coo = map_coo.reshape(2, -1)

    # coo转为csc
    sparse_tensor = SparseTensor.from_edge_index(
        (torch.from_numpy(map_coo_row), torch.from_numpy(map_coo_col)),
        sparse_sizes=(num_nodes, num_nodes),
    )
    indptr, indices, _ = sparse_tensor.csc()
    assert indptr.shape[0] == num_nodes + 1, f"{indptr.shape[0]} != {num_nodes + 1}"
    indptr = indptr.numpy()
    indices = indices = indices.numpy()

    # 写入文件
    indptr.tofile(os.path.join(out_path, "indptr.bin"))
    indices.tofile(os.path.join(out_path, "indices.bin"))
    map_coo.tofile(os.path.join(out_path, "coo.bin"))
    map_feat.tofile(os.path.join(out_path, "feat.bin"))
    map_labels.tofile(os.path.join(out_path, "labels.bin"))
    map_train_idx.tofile(os.path.join(out_path, "train_idx.bin"))
    map_val_idx.tofile(os.path.join(out_path, "val_idx.bin"))
    map_test_idx.tofile(os.path.join(out_path, "test_idx.bin"))
    np.savetxt(os.path.join(out_path, "offsets.txt"), np_offsets, fmt="%d")
    config = {
        "num_nodes": num_nodes,
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
    }
    with open(os.path.join(out_path, "conf.json"), "w") as f:
        json.dump(config, f)


def mapping(coo_row, coo_col, nodes_part, feat, labels, train_idx, val_idx, test_idx):
    all_nodes = np.concatenate(nodes_part)
    map_id = np.arange(all_nodes.shape[0], dtype=np.int64)
    global_to_local_map = -np.ones(all_nodes.shape[0], dtype=np.int64)
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


def test():
    adjacency_list = [
        np.array([4, 2, 1]),
        np.array([0, 2, 3]),
        np.array([4, 3, 1, 0]),
        np.array([1, 2, 5, 6]),
        np.array([0, 2, 5]),
        np.array([4, 3, 6]),
        np.array([5, 3]),
    ]
    coo_row = []
    coo_col = []
    for i, l in enumerate(adjacency_list):
        coo_row.extend([i] * len(l))
        coo_col.extend(list(l))
    coo_row = np.array(coo_row, dtype=np.int64)
    coo_col = np.array(coo_col, dtype=np.int64)
    print(coo_row)
    print(coo_col)
    n_cuts, membership = pymetis.part_graph(2, adjacency=adjacency_list)
    # n_cuts = 3
    # membership = [1, 1, 1, 0, 1, 0, 0]
    print("#n_cuts: ", n_cuts)
    nodes_part = []
    for i in range(2):
        part = np.argwhere(np.array(membership) == i).ravel()
        nodes_part.append(part)
    print(nodes_part)
    feat = np.arange(7, dtype=np.float32)
    feat = feat.reshape(-1, 1)
    feat = feat.repeat(5, 1)
    labels = np.arange(7)
    labels = labels.reshape(-1, 1)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(current_dir, "data/test")
    __process_data(coo_row, coo_col, feat, labels, 2, out_path)


def generate_test_feat(out_path, feat_dim, num_nodes):
    feat = np.arange(num_nodes, dtype=np.float32)
    feat = feat.reshape(-1, 1)
    feat = feat.repeat(feat_dim, 1)
    feat.tofile(os.path.join(out_path, "feat.bin"))
    config = {"num_nodes": num_nodes, "feat_dim": feat_dim}
    with open(os.path.join(out_path, "conf.json"), "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-papers100M",
        help="dataset name: ogbn-products, ogbn-papers100M",
    )
    argparser.add_argument("--data-path", type=str, default="/home/ningxin/data")
    argparser.add_argument("--num-parts", type=int, default=8)
    argparser.add_argument(
        "--part-method", type=str, default="metis", help="`metis` or `dne`"
    )
    argparser.add_argument(
        "--dne-part-path",
        type=str,
        default="/home/ningxin/data/ogbn-products-dne-part8",
        help="path of dne partitions",
    )
    args = argparser.parse_args()
    # test()
    # process_dgl_data(args)
    # generate_test_feat('./data/test_feat', 100, 32456)
    if args.part_method == "metis":
        process_dgl_data(args)
    elif args.part_method == "dne":
        process_products_with_dne(args, dne_part_path=args.dne_part_path)
    else:
        raise RuntimeError("Unsupported partition method: {}".format(args.method))
