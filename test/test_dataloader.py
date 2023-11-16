import time
import os
import sys
import json

import torch as th
import dgl
import tqdm
from torch_sparse import SparseTensor

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from sinfer.dataloader import DataLoader, PygDataLoader
from sinfer.data import SinferDataset
from sinfer.cpp_core import tensor_free


def test():
    test_path = "/workspace/ningxin/data/ogbn-products-ssd-infer"
    # test_path = '/workspace/ningxin/ssdgnn/ssdgnn/sinfer/data/test_feat'
    with open(os.path.join(test_path, "conf.json"), "r") as f:
        config = json.load(f)
    feat_dim = config["feat_dim"]
    num_nodes = config["num_nodes"]
    data_path = os.path.join(test_path, "feat.bin")
    batch_size = 1000
    kwargs = {"batch_size": batch_size, "drop_last": False}
    coo_src = th.randint(0, num_nodes, (2 * num_nodes,), dtype=th.int64)
    coo_dst = th.randint(0, num_nodes, (2 * num_nodes,), dtype=th.int64)
    graph = dgl.graph((coo_src, coo_dst))
    offsets = [0, 5000, 10000, 15000, 20000, 25000, 30000, 32456]
    infer_dataloader = DataLoader(
        graph, data_path, feat_dim, offsets, prefetch=True, **kwargs
    )
    for i in range(2):
        start = time.time()
        num_nodes = 0
        for step, (input_nodes, seeds, blocks) in enumerate(infer_dataloader):
            x = blocks[0].srcdata["feat"]
            # print(x)
            # print(seeds)
            num_nodes += seeds.shape[0]
        print("total time: {}, num nodes: {}".format(time.time() - start, num_nodes))
    infer_dataloader.shutdown()
    print("shutdown")


def test_products():
    data = SinferDataset("/home/data/ogbn-products-ssd-infer")
    coo_row, coo_col = data.coo()
    graph = dgl.graph((coo_row, coo_col))
    print(graph)
    kwargs = {
        "batch_size": 1000,
        "drop_last": False,
    }
    infer_dataloader = DataLoader(
        graph, data.feat_path, data.feat_dim, data.offsets, prefetch=True, **kwargs
    )
    start = time.time()
    num_nodes = 0
    for input_nodes, seeds, blocks in tqdm.tqdm(infer_dataloader):
        x = blocks[0].srcdata["feat"]
        num_nodes += input_nodes.shape[0]
        # tensor_free(x)
    print("total time: {}, num nodes: {}".format(time.time() - start, num_nodes))


def get_csc(data):
    coo_row, coo_col = data.coo()
    sparse_tensor = SparseTensor.from_edge_index((coo_row, coo_col))
    indptr, indices, _ = sparse_tensor.csc()
    return indptr, indices


def test_pyg_products():
    data = SinferDataset("/home/data/ogbn-products-ssd-infer")
    indptr, indices = get_csc(data)
    kwargs = {"batch_size": 1000, "drop_last": False, "num_workers": 0}
    offsets = [0, data.num_nodes]
    infer_dataloader = PygDataLoader(
        indptr, indices, data.feat_path, data.feat_dim, offsets, prefetch=True, **kwargs
    )
    start = time.time()
    num_nodes = 0
    for batch_size, seeds, adjs, feat in tqdm.tqdm(infer_dataloader):
        x = feat
        num_nodes += seeds.shape[0]
        # tensor_free(x)
    print("total time: {}, num nodes: {}".format(time.time() - start, num_nodes))


if __name__ == "__main__":
    os.environ["SINFER_NUM_THREADS"] = "16"
    # test_products()
    test_pyg_products()
