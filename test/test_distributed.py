import numpy as np
import torch
import os
import sys
import torch.multiprocessing as mp
import logging

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from sinfer.distributed import PartInfo
import sinfer.distributed as dist

logger = logging.getLogger()
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def test_part_info():
    local_nodes = torch.arange(5, dtype=torch.int64)
    global_nodes = torch.tensor([0, 2, 4, 5, 6], dtype=torch.int64)
    local_degree = torch.tensor([1, 2, 2, 2, 3], dtype=torch.int64)
    global_degree = torch.tensor([2, 2, 2, 4, 3], dtype=torch.int64)
    part_info = PartInfo(local_nodes, global_nodes, local_degree, global_degree)
    nodes = torch.tensor([2, 1, 0, 3, 4], dtype=torch.int64)
    # [2, 3], [0, 3]
    index, boundary_nodes = part_info.get_boundary_nodes(nodes)
    np.testing.assert_array_equal(index.numpy(), np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(
        boundary_nodes.numpy(), np.array([0, 3], dtype=np.int64)
    )
    # [0, 5]
    global_boundary_nodes = part_info.local_to_global_id(boundary_nodes)
    np.testing.assert_array_equal(
        global_boundary_nodes.numpy(), np.array([0, 5], dtype=np.int64)
    )

    # [0, 3]
    all_boundary_nodes = part_info.get_all_boundary_nodes()
    np.testing.assert_array_equal(
        all_boundary_nodes.numpy(), np.array([0, 3], dtype=np.int64)
    )

    # [0, 1, 4], [2, 1, 4]
    index, inner_nodes = part_info.get_inner_nodes(nodes)
    np.testing.assert_array_equal(index.numpy(), np.array([0, 1, 4], dtype=np.int64))
    np.testing.assert_array_equal(
        inner_nodes.numpy(), np.array([2, 1, 4], dtype=np.int64)
    )
    # [4, 2, 6]
    global_inner_nodes = part_info.local_to_global_id(inner_nodes)
    np.testing.assert_array_equal(
        global_inner_nodes.numpy(), np.array([4, 2, 6], dtype=np.int64)
    )

    # [1, 2, 4]
    inner_nodes = part_info.get_all_inner_nodes()
    np.testing.assert_array_equal(
        inner_nodes.numpy(), np.array([1, 2, 4], dtype=np.int64)
    )


def test_gas_store_sum(rank, world_size):
    num_total_nodes = 5
    num_hiddens = 10
    emb = torch.ones(4, num_hiddens, dtype=torch.float32)
    if rank == 0:
        x = torch.tensor([3, 1, 1, 1], dtype=torch.float32).view(4, -1)
    else:
        x = torch.tensor([2, 2, 3, 1], dtype=torch.float32).view(4, -1)
    emb = emb * x
    y = torch.ones(num_total_nodes, num_hiddens, dtype=torch.float32) * torch.tensor(
        [3, 3, 3, 3, 2], dtype=torch.float32
    ).view(num_total_nodes, -1)
    test_gas_store(rank, world_size, "sum", emb, y)


def test_gas_store_mean(rank, world_size):
    num_total_nodes = 5
    num_hiddens = 10
    emb = torch.ones(4, num_hiddens, dtype=torch.float32)
    y = torch.ones(num_total_nodes, num_hiddens, dtype=torch.float32)
    test_gas_store(rank, world_size, "mean", emb, y)


def test_gas_store_max(rank, world_size):
    num_total_nodes = 5
    num_hiddens = 10
    emb = torch.ones(4, num_hiddens, dtype=torch.float32)
    y = torch.ones(num_total_nodes, num_hiddens, dtype=torch.float32)
    test_gas_store(rank, world_size, "max", emb, y)


def test_gas_store_min(rank, world_size):
    num_total_nodes = 5
    num_hiddens = 10
    emb = torch.ones(4, num_hiddens, dtype=torch.float32)
    y = torch.ones(num_total_nodes, num_hiddens, dtype=torch.float32)
    test_gas_store(rank, world_size, "min", emb, y)


def test_gas_store(rank, world_size, reduce, emb, y):
    # part0 edges: [(0,1), (0,2), (0,4)]
    # part1 edges: [(1,2), (1,3), (2,3), (3,4)]
    num_total_nodes = 5
    num_hiddens = 10
    if rank == 0:
        local_nodes = torch.arange(4, dtype=torch.int64)
        global_nodes = torch.tensor([0, 1, 2, 4], dtype=torch.int64)
        local_degree = torch.tensor([3, 1, 1, 1], dtype=torch.int64)
        global_degree = torch.tensor([3, 3, 3, 2], dtype=torch.int64)
    else:
        local_nodes = torch.arange(4, dtype=torch.int64)
        global_nodes = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        local_degree = torch.tensor([2, 2, 3, 1], dtype=torch.int64)
        global_degree = torch.tensor([3, 3, 3, 2], dtype=torch.int64)
        part_info = PartInfo(local_nodes, global_nodes, local_degree, global_degree)
    part_info = PartInfo(local_nodes, global_nodes, local_degree, global_degree)
    dist.init_master_store(rank, world_size, part_info)

    gas_store = dist.GASStore(
        (num_total_nodes // world_size + 1, num_hiddens),
        dtype=torch.float32,
        name=f"gas_store_{reduce}",
        reduce=reduce,
    )
    node_list = [torch.tensor([i], dtype=torch.int64) for i in range(4)]
    for node in node_list:
        gas_store.scatter(node, emb[node])
    gas_store.sync()
    gas_store.apply_all(emb)
    gas_store.gather_all(emb)
    print(f"{rank}, data={emb.numpy()}")
    dist.shutdown()
    np.testing.assert_array_equal(emb.numpy(), y[global_nodes].numpy())


def run_test_gas_store():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(test_gas_store_max, args=(world_size,), nprocs=world_size, join=True)
    mp.spawn(test_gas_store_mean, args=(world_size,), nprocs=world_size, join=True)
    mp.spawn(test_gas_store_sum, args=(world_size,), nprocs=world_size, join=True)
    mp.spawn(test_gas_store_min, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    for i in range(10):
        test_part_info()
        run_test_gas_store()
