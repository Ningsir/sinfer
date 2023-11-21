import numpy as np
import torch as th
import time
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from sinfer.store import FeatureStore, EmbeddingStore
from sinfer.data import SinferDataset


def __test_read_base(data_path, prefetch):
    data = SinferDataset(data_path)
    feat_path = data.feat_path
    feat_dim = data.feat_dim
    offsets = data.offsets

    raw_data = np.fromfile(feat_path, dtype=np.float32).reshape(-1, feat_dim)
    store = FeatureStore(
        data.feat_path,
        offsets,
        data.num_nodes,
        feat_dim,
        prefetch=prefetch,
        dtype=th.float32,
    )
    print(id(FeatureStore))
    size = 20000
    nodes = th.randint(1, data.num_nodes - 1, (size,), dtype=th.int64)

    total = nodes.shape[0] * feat_dim * 4
    t1 = time.time()
    f1 = raw_data[nodes.numpy()]
    mem_time = time.time() - t1
    speed1 = total / (mem_time * 1e9)

    num_parts = len(offsets) - 1
    for i in range(num_parts):
        # 将第i个分区加载到缓存中
        store.update_cache(i)
        t2 = time.time()
        f2 = store.gather(nodes)
        sinfer_mem_time = time.time() - t2
        speed2 = total / (sinfer_mem_time * 1e9)
        np.testing.assert_array_equal(f1, f2.numpy())
        print(
            "cache: {}, mem speed: {} GB/s, sinfer speed: {} GB/s".format(
                i, speed1, speed2
            )
        )


def test_read_prefetch(data_path):
    __test_read_base(data_path, prefetch=True)


def test_read_not_prefetch(data_path):
    __test_read_base(data_path, prefetch=False)


def __test_write_base(
    data,
    offsets,
    num_nodes,
    feat_dim,
    prefetch,
    num_workers=1,
    batch_size=1000,
    writer_seq=True,
):
    tmp_path = "embedding.bin"

    # 1. 顺序写入batch_size的数据
    emb_store = EmbeddingStore(
        tmp_path,
        offsets,
        num_nodes,
        feat_dim,
        prefetch=prefetch,
        dtype=th.float32,
        num_writer_workers=num_workers,
        writer_seq=writer_seq,
    )
    for i in range(0, num_nodes, batch_size):
        if num_nodes - i < batch_size:
            start, end = i, num_nodes
        else:
            start, end = i, i + batch_size
        x = data[start:end, :]
        nodes = th.arange(start, end, dtype=th.int64)
        emb_store.write_data(nodes, x)
    emb_store.flush()
    # 2. 读取
    size = 20000
    nodes = th.randint(1, num_nodes - 1, (size,), dtype=th.int64)

    total = nodes.shape[0] * feat_dim * 4
    t1 = time.time()
    f1 = data[nodes.numpy()]
    mem_time = time.time() - t1
    speed1 = total / (mem_time * 1e9)

    num_parts = len(offsets) - 1
    for i in range(num_parts):
        # 将第i个分区加载到缓存中
        emb_store.update_cache(i)
        t2 = time.time()
        f2 = emb_store.gather(nodes)
        sinfer_mem_time = time.time() - t2
        speed2 = total / (sinfer_mem_time * 1e9)
        np.testing.assert_array_equal(f1, f2.numpy())
        print(
            "cache: {}, mem speed: {} GB/s, sinfer speed: {} GB/s".format(
                i, speed1, speed2
            )
        )


def __test_write1_base(num_writer_workers, writer_seq):
    num_nodes = 234536
    feat_dim = 128
    offsets = [0, 12345, 67890, 102341, 160012, 234536]
    prefetch = True

    data = th.randn((num_nodes, feat_dim), dtype=th.float32)
    __test_write_base(
        data,
        offsets,
        num_nodes,
        feat_dim,
        prefetch,
        batch_size=1000,
        writer_seq=writer_seq,
        num_workers=num_writer_workers,
    )


def test_write1():
    __test_write1_base(1, True)
    __test_write1_base(4, True)
    __test_write1_base(1, False)
    __test_write1_base(4, False)


if __name__ == "__main__":
    os.environ["SINFER_NUM_THREADS"] = "16"
    data_path = "/home/ningxin/data/ogbn-products-ssd-infer"
    # test_read_not_prefetch(data_path)
    # for _ in range(10):
    #     test_read_prefetch(data_path)
    for _ in range(10):
        test_write1()
