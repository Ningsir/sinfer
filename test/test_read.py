import numpy as np
import torch as th
import sys
import os
import json

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from sinfer.cpp_core import gather_mem, gather_sinfer

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


def test_gather_mem_base(data_path, start, end, feat_dim):
    raw_data = np.fromfile(data_path, dtype=np.float32)
    raw_data = raw_data.reshape(-1, feat_dim)
    
    nodes = np.arange(start, end, dtype=np.int64)
    feat1 = raw_data[nodes]
    
    feat2 = gather_mem(data_path, start, end, feat_dim)
    
    np.testing.assert_allclose(feat1, feat2.numpy())


def test_gather_mem_products():
    test_path = '/workspace/ningxin/data/ogbn-products-ssd-infer'
    with open(os.path.join(test_path, 'conf.json'), 'r') as f:
        config = json.load(f)
    feat_dim = config['feat_dim']
    num_nodes = config['num_nodes']
    data_path = os.path.join(test_path, "feat.bin")
    test_gather_mem_base(data_path, 0, num_nodes, feat_dim)
    test_gather_mem_base(data_path, 0, num_nodes // 2, feat_dim)
    test_gather_mem_base(data_path, num_nodes // 2, num_nodes, feat_dim)


def test_gather_sinfer_base(data_path, cache_start, cache_end, feat_dim, num_nodes):
    
    raw_data = np.fromfile(data_path, dtype=np.float32)
    raw_data = raw_data.reshape(-1, feat_dim)
    cache = gather_mem(data_path, cache_start, cache_end, feat_dim)
    
    nodes = th.randint(0, num_nodes, (num_nodes // 10, ), dtype=th.int64)

    feat1 = raw_data[nodes.numpy()]
    # print(feat1)
    feat = gather_sinfer(data_path, nodes, feat_dim, cache, cache_start, cache_end)
    # print(feat)
    np.testing.assert_allclose(feat1, feat.numpy())


def test_gather_sinfer_test_data():
    test_path = '/workspace/ningxin/ssdgnn/ssdgnn/sinfer/data/test_feat'
    with open(os.path.join(test_path, 'conf.json'), 'r') as f:
        config = json.load(f)
    feat_dim = config['feat_dim']
    num_nodes = config['num_nodes']
    data_path = os.path.join(test_path, "feat.bin")
    test_gather_sinfer_base(data_path, 0, num_nodes, feat_dim, num_nodes)
    test_gather_sinfer_base(data_path, 0, 0, feat_dim, num_nodes)
    test_gather_sinfer_base(data_path, num_nodes, num_nodes, feat_dim, num_nodes)
    test_gather_sinfer_base(data_path, 100, num_nodes - 100, feat_dim, num_nodes)
    test_gather_sinfer_base(data_path, 1000, num_nodes - 1000, feat_dim, num_nodes)


if __name__ == "__main__":
    # test_gather_sinfer_test_data()
    test_gather_mem_products()
