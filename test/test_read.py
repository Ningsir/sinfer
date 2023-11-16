import numpy as np
import torch as th
import sys
import os
import json
import time

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from sinfer.cpp_core import gather_mem, gather_sinfer, gather_ssd, gather_sinfer1
from sinfer.data import SinferDataset


def generate_test_feat(out_path, feat_dim, num_nodes):
    feat = np.arange(num_nodes, dtype=np.float32)
    feat = feat.reshape(-1, 1)
    feat = feat.repeat(feat_dim, 1)
    feat.tofile(os.path.join(out_path, "feat.bin"))
    config = {"num_nodes": num_nodes, "feat_dim": feat_dim}
    with open(os.path.join(out_path, "conf.json"), "w") as f:
        json.dump(config, f)


def test_gather_mem_base(data_path, start, end, feat_dim):
    raw_data = np.fromfile(data_path, dtype=np.float32)
    raw_data = raw_data.reshape(-1, feat_dim)

    nodes = np.arange(start, end, dtype=np.int64)
    feat1 = raw_data[nodes]

    feat2 = gather_mem(data_path, start, end, feat_dim)

    np.testing.assert_allclose(feat1, feat2.numpy())


def test_gather_mem_products():
    test_path = "/workspace/ningxin/data/ogbn-products-ssd-infer"
    with open(os.path.join(test_path, "conf.json"), "r") as f:
        config = json.load(f)
    feat_dim = config["feat_dim"]
    num_nodes = config["num_nodes"]
    data_path = os.path.join(test_path, "feat.bin")
    test_gather_mem_base(data_path, 0, num_nodes, feat_dim)
    test_gather_mem_base(data_path, 0, num_nodes // 2, feat_dim)
    test_gather_mem_base(data_path, num_nodes // 2, num_nodes, feat_dim)


def test_gather_sinfer_base(data_path, cache_start, cache_end, feat_dim, num_nodes):
    raw_data = np.fromfile(data_path, dtype=np.float32)
    raw_data = raw_data.reshape(-1, feat_dim)
    cache = gather_mem(data_path, cache_start, cache_end, feat_dim)

    nodes = th.randint(0, num_nodes, (num_nodes // 10,), dtype=th.int64)

    feat1 = raw_data[nodes.numpy()]
    # print(feat1)
    feat = gather_sinfer(data_path, nodes, feat_dim, cache, cache_start, cache_end)
    # print(feat)
    np.testing.assert_allclose(feat1, feat.numpy())


def test_gather_sinfer_test_data():
    test_path = "/workspace/ningxin/ssdgnn/ssdgnn/sinfer/data/test_feat"
    with open(os.path.join(test_path, "conf.json"), "r") as f:
        config = json.load(f)
    feat_dim = config["feat_dim"]
    num_nodes = config["num_nodes"]
    data_path = os.path.join(test_path, "feat.bin")
    test_gather_sinfer_base(data_path, 0, num_nodes, feat_dim, num_nodes)
    test_gather_sinfer_base(data_path, 0, 0, feat_dim, num_nodes)
    test_gather_sinfer_base(data_path, num_nodes, num_nodes, feat_dim, num_nodes)
    test_gather_sinfer_base(data_path, 100, num_nodes - 100, feat_dim, num_nodes)
    test_gather_sinfer_base(data_path, 1000, num_nodes - 1000, feat_dim, num_nodes)


def test_mem_gather_speed(data_path):
    data = SinferDataset(data_path)
    feat_path = data.feat_path
    feat_dim = data.feat_dim
    offsets = data.offsets
    start, end = int(offsets[2]), int(offsets[3])
    cache = gather_mem(feat_path, start, end, feat_dim)
    raw_data = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)

    size = (end - start) * 1
    nodes = th.randint(start, end, (size,), dtype=th.int64)
    total = nodes.shape[0] * feat_dim * 4
    t1 = time.time()
    f1 = raw_data[nodes.numpy()]
    mem_time = time.time() - t1
    speed1 = total / (mem_time * 1e9)
    t2 = time.time()
    f2 = gather_sinfer(feat_path, nodes, feat_dim, cache, start, end)
    sinfer_mem_time = time.time() - t2
    speed2 = total / (sinfer_mem_time * 1e9)
    np.testing.assert_allclose(f1, f2.numpy())
    print(
        "mem gather test, mem speed: {} GB/s, sinfer cache speed: {} GB/s".format(
            speed1, speed2
        )
    )


def test_cache_gather_speed(data_path):
    data = SinferDataset(data_path)
    feat_path = data.feat_path
    feat_dim = data.feat_dim
    offsets = data.offsets
    start, end = int(offsets[2]), int(offsets[3])
    cache = gather_mem(feat_path, start, end, feat_dim)
    raw_data = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)

    size1 = end - start
    size2 = size1 // 20
    # 内顶点
    nodes1 = th.randint(start, end, (size1,), dtype=th.int64)
    # 外顶点
    nodes2 = th.randint(end, int(offsets[-1]), (size2,), dtype=th.int64)
    nodes = th.concat((nodes1, nodes2))
    shuffle_idx = th.randperm(nodes.shape[0])
    nodes = nodes[shuffle_idx]
    total = nodes.shape[0] * feat_dim * 4
    t1 = time.time()
    f1 = raw_data[nodes.numpy()]
    mem_time = time.time() - t1
    speed1 = total / (mem_time * 1e9)
    t2 = time.time()
    f2 = gather_sinfer1(feat_path, nodes, feat_dim, cache, start, end)
    sinfer_mem_time = time.time() - t2
    speed2 = total / (sinfer_mem_time * 1e9)
    np.testing.assert_allclose(f1, f2.numpy())
    print(
        "cache gather test, mem speed: {} GB/s, sinfer cache speed: {} GB/s".format(
            speed1, speed2
        )
    )


def test_ssd_direct_gather_speed(data_path):
    data = SinferDataset(data_path)
    feat_path = data.feat_path
    feat_dim = data.feat_dim
    offsets = data.offsets
    start, end = 0, 0
    cache = gather_mem(feat_path, start, end, feat_dim)
    raw_data = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)

    size = 20000
    nodes = th.randint(0, data.num_nodes, (size,), dtype=th.int64)

    total = nodes.shape[0] * feat_dim * 4
    t1 = time.time()
    f1 = raw_data[nodes.numpy()]
    mem_time = time.time() - t1
    speed1 = total / (mem_time * 1e9)
    t2 = time.time()
    f2 = gather_sinfer(feat_path, nodes, feat_dim, cache, start, end)
    sinfer_mem_time = time.time() - t2
    speed2 = total / (sinfer_mem_time * 1e9)
    np.testing.assert_allclose(f1, f2.numpy())
    print(
        "ssd gather test, mem speed: {} GB/s, sinfer speed: {} GB/s".format(
            speed1, speed2
        )
    )


def test_full_ssd_no_direct_gather_speed(data_path):
    data = SinferDataset(data_path)
    feat_path = data.feat_path
    feat_dim = data.feat_dim
    raw_data = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)

    size = 20000
    nodes = th.randint(0, data.num_nodes, (size,), dtype=th.int64)

    total = nodes.shape[0] * feat_dim * 4
    t1 = time.time()
    f1 = raw_data[nodes.numpy()]
    mem_time = time.time() - t1
    speed1 = total / (mem_time * 1e9)
    t2 = time.time()
    f2 = gather_ssd(feat_path, nodes, feat_dim)
    sinfer_mem_time = time.time() - t2
    speed2 = total / (sinfer_mem_time * 1e9)

    sort_nodes, _ = th.sort(nodes)
    t3 = time.time()
    f3 = gather_ssd(feat_path, sort_nodes, feat_dim)
    sort_sinfer_time = time.time() - t3
    speed3 = total / (sort_sinfer_time * 1e9)

    np.testing.assert_allclose(f1, f2.numpy())
    print(
        "full ssd gather test, mem speed: {} GB/s, sinfer gather ssd speed: {} GB/s, sort sinfer ssd speed: {} GB/s".format(
            speed1, speed2, speed3
        )
    )


if __name__ == "__main__":
    # test_gather_sinfer_test_data()
    # test_gather_mem_products()
    os.environ["SINFER_NUM_THREADS"] = "16"
    data_path = "/home/ningxin/data/ogbn-products-ssd-infer"
    test_mem_gather_speed(data_path)
    test_cache_gather_speed(data_path)
    test_ssd_direct_gather_speed(data_path)
    test_full_ssd_no_direct_gather_speed(data_path)
