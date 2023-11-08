import pymetis
import argparse
from ogb.nodeproppred import DglNodePropPredDataset
import scipy
import numpy as np
import json
import torch
import os

from sinfer.utils import coo_to_adj_list


def process_dgl_data(args):
    out_path = os.path.join(args.data_path, args.dataset + "-ssd-infer")
    num_parts = args.num_parts
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
    __process_data(src, dst, nfeat, labels, num_parts, out_path)


def __process_data(coo_row, coo_col, feat, labels, num_parts, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    adj = coo_to_adj_list(coo_row, coo_col)
    
    n_cuts, membership = pymetis.part_graph(num_parts, adj)
    
    print("#n_cuts: {}".format(n_cuts))
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
    # 按照nodes_part中的顺序对顶点ID进行重映射
    map_coo_row, map_coo_col, map_feat, map_labels = mapping(coo_row, coo_col, nodes_part, feat, labels)
    map_coo = np.concatenate((map_coo_row, map_coo_col))
    map_coo = map_coo.reshape(2, -1)
    
    # 写入文件
    map_coo.tofile(os.path.join(out_path, "coo.bin"))
    map_feat.tofile(os.path.join(out_path, "feat.bin"))
    map_labels.tofile(os.path.join(out_path, "labels.bin"))
    np.savetxt(os.path.join(out_path, "offsets.txt"), np_offsets, fmt="%d")


def mapping(coo_row, coo_col, nodes_part, feat, labels):
    all_nodes = np.concatenate(nodes_part)
    map_id = np.arange(all_nodes.shape[0], dtype=np.int64)
    global_to_local_map = -np.ones(all_nodes.shape[0], dtype=np.int64)
    global_to_local_map[all_nodes] = map_id
    map_feat = feat[all_nodes]
    map_labels = labels[all_nodes]
    
    # global to local
    map_coo_row = global_to_local_map[coo_row]
    map_coo_col = global_to_local_map[coo_col]
    print(map_coo_row)
    print(map_coo_col)
    return map_coo_row, map_coo_col, map_feat, map_labels


def test():
    adjacency_list = [np.array([4, 2, 1]),
                  np.array([0, 2, 3]),
                  np.array([4, 3, 1, 0]),
                  np.array([1, 2, 5, 6]),
                  np.array([0, 2, 5]),
                  np.array([4, 3, 6]),
                  np.array([5, 3])]
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


def test_read(data_path):
    pass

def generate_test_feat(out_path, feat_dim, num_nodes):
    feat = np.arange(num_nodes, dtype=np.float32)
    feat = feat.reshape(-1, 1)
    feat = feat.repeat(feat_dim, 1)
    feat.tofile(os.path.join(out_path, "feat.bin"))
    config = {
        "num_nodes": num_nodes,
        "feat_dim": feat_dim
    }
    with open(os.path.join(out_path, "conf.json"), 'w') as f:
        json.dump(config, f)
    

if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--data-path', type=str, default='/workspace/ningxin/data')
    argparser.add_argument('--num-parts', type=int, default=4)
    
    args = argparser.parse_args()
    test()
    # process_dgl_data(args)
    # generate_test_feat('./data/test_feat', 100, 32456)
