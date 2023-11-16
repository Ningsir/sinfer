import time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch as th
import dgl

from sinfer.data import SinferDataset
from sinfer.dataloader import DataLoader
from sinfer.cpp_core import gather_mem, gather_sinfer


# 测试采样结果中外顶点和内顶点占比
def test_sample_outing_nodes(data_path, num_layers=1):
    device = th.device("cuda:0")
    data = SinferDataset(data_path)
    coo_row, coo_col = data.coo()
    graph = dgl.graph((coo_row, coo_col), device=device)
    print(graph)
    kwargs = {
        "batch_size": 1000,
        "drop_last": False,
    }
    offsets = data.offsets
    left, right = offsets[1], offsets[2]
    nodes = th.arange(left, right, dtype=th.int64).to(device)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
    infer_dataloader = dgl.dataloading.DataLoader(graph, nodes, sampler, **kwargs)
    start = time.time()
    inner_nodes = 0
    out_nodes = 0
    for step, (input_nodes, seeds, blocks) in enumerate(infer_dataloader):
        i_nodes = input_nodes[(input_nodes >= left) & (input_nodes < right)]
        o_nodes = input_nodes[(input_nodes < left) | (input_nodes >= right)]
        assert i_nodes.shape[0] + o_nodes.shape[0] == input_nodes.shape[0]
        inner_nodes += i_nodes.shape[0]
        out_nodes += o_nodes.shape[0]
        # print("nodes: {}, innet nodes: {}, out nodes: {}".format(input_nodes.shape[0], i_nodes.shape[0], o_nodes.shape[0]))
    print(
        "total time: {}, inner nodes: {}, out nodes: {}".format(
            time.time() - start, inner_nodes, out_nodes
        )
    )


if __name__ == "__main__":
    data_path = "/workspace/ningxin/data/ogbn-products-ssd-infer"
    test_sample_outing_nodes(data_path, num_layers=3)
