import logging
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef, rpc_sync, rpc_async
from typing import List, Union
from threading import Lock
import copy

from .part_info import PartInfo

LOCAL_MASTER_STORE_RREF = None
MASTER_STORE_RREF_DICT = {}
global_lock = Lock()


def _call_method(method, rref: RRef, *args, **kwargs):
    r"""
    a helper function to call a method on the given RRef
    """
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref: RRef, *args, **kwargs):
    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _set_master_store_dict(name, rref):
    with global_lock:
        global MASTER_STORE_RREF_DICT
        MASTER_STORE_RREF_DICT[name] = rref


def _get_master_store_name(rank):
    return f"__master_store_{rank}"


def _get_master_store_rref(rank):
    return MASTER_STORE_RREF_DICT[_get_master_store_name(rank)]


def _get_local_master_store_rref():
    return LOCAL_MASTER_STORE_RREF


def _get_master_store_rref_dict():
    return MASTER_STORE_RREF_DICT


def init_master_store(rank, world_size, part_info):
    """
    初始化`master_store`以及分布式RPC模块
    """
    name = _get_master_store_name(rank)
    rpc.init_rpc(name, rank=rank, world_size=world_size)
    name_list = []
    for i in range(world_size):
        name_list.append(_get_master_store_name(i))
    global LOCAL_MASTER_STORE_RREF
    LOCAL_MASTER_STORE_RREF = RRef(MasterStore(rank, world_size, part_info))
    for store_name in name_list:
        rpc.rpc_sync(
            store_name, _set_master_store_dict, args=(name, LOCAL_MASTER_STORE_RREF)
        )
    logging.info(f"Initialized master store for rank {rank}")


def shutdown():
    """
    清除`master_store`以及关闭分布式RPC模块
    """
    rpc.shutdown()
    global LOCAL_MASTER_STORE_RREF
    LOCAL_MASTER_STORE_RREF = None
    global MASTER_STORE_RREF_DICT
    MASTER_STORE_RREF_DICT = {}
    logging.info(f"shutwodn master store")


def get_rank():
    assert (
        LOCAL_MASTER_STORE_RREF is not None
    ), "Distributed module is not initialized. Please call sinfer.distributed.init_master_store."
    return MasterStore.rank


def get_world_size():
    assert (
        LOCAL_MASTER_STORE_RREF is not None
    ), "Distributed module is not initialized. Please call sinfer.distributed.init_master_store."
    return MasterStore.world_size


def get_part_info():
    assert (
        LOCAL_MASTER_STORE_RREF is not None
    ), "Distributed module is not initialized. Please call sinfer.distributed.init_master_store."
    return MasterStore.part_info


class MasterStore:
    """
    存储master节点的embedding, 并处理`GASStore`发起的`scatter`, `apply`, `gather`等RPC请求.
    每个worker存在唯一的单例`MasterStore`
    """

    rank = -1
    world_size = -1
    part_info = None

    def __init__(self, rank, world_size, part_info: PartInfo):
        MasterStore.rank = rank
        MasterStore.world_size = world_size
        MasterStore.part_info = part_info
        # {name: torch.Tensor}
        self.data_ = {}
        self.scatter_lock = Lock()
        self.apply_lock = Lock()

    def init_data(self, name: str, shape: List[int], dtype: torch.dtype, reduce="mean"):
        if name in self.data_.keys():
            raise Exception("data name={name} already exists")
        if reduce == "max":
            self.data_[name] = torch.full(shape, torch.finfo(dtype).min, dtype=dtype)
        elif reduce == "min":
            self.data_[name] = torch.full(shape, torch.finfo(dtype).max, dtype=dtype)
        else:
            self.data_[name] = torch.zeros(shape, dtype=dtype)
        logging.info(
            f"master store init data name={name}, shape={shape}, dtype={dtype}, reduce={reduce}"
        )

    def delete_data(self, name):
        if name not in self.data_.keys():
            raise Exception("data name={name} does not exist")
        del self.data_[name]
        logging.info(f"master store delete data name={name}")

    def scatter(self, data_name, global_nodes, x, reduce="mean"):
        assert reduce in [
            "mean",
            "sum",
            "max",
            "min",
        ], f"reduce={reduce} not supported!"
        if data_name not in self.data_.keys():
            raise Exception("data name={name} does not exist")
        # 1. 接受来自mirror节点的emb
        indices = torch.div(global_nodes, MasterStore.world_size, rounding_mode="floor")
        with self.scatter_lock:
            if reduce == "sum" or reduce == "mean":
                self.data_[data_name][indices] += x
            elif reduce == "max":
                self.data_[data_name][indices] = torch.max(
                    self.data_[data_name][indices], x
                )
            else:
                self.data_[data_name][indices] = torch.min(
                    self.data_[data_name][indices], x
                )
        # TODO：num_mirror_nodes对应nodes减1，如果等于0，则表示可以执行apply

    def apply(self, data_name, global_nodes, x, nonlinear_func=None):
        """
        对master节点应用非线性激活函数, 执行更新
        """
        if data_name not in self.data_.keys():
            raise Exception("data name={name} does not exist")
        if nonlinear_func is None:
            return
        indices = torch.div(global_nodes, MasterStore.world_size, rounding_mode="floor")
        x = nonlinear_func(self.data_[data_name][indices])
        self.data_[data_name][indices] += x

    def apply_all(self, data_name, nonlinear_func=None):
        """
        对所有master节点应用非线性激活函数, 执行更新
        """
        if data_name not in self.data_.keys():
            raise Exception("data name={name} does not exist")
        if nonlinear_func is None:
            return
        with self.apply_lock:
            self.data_[data_name] = nonlinear_func(self.data_[data_name])

    def gather(self, data_name, global_nodes):
        indices = torch.div(global_nodes, MasterStore.world_size, rounding_mode="floor")
        return self.data_[data_name][indices]


class GASStore:
    """
    和`MasterStore`进行通信, 提供`scatter`, `apply`, `gather`等API
    """

    def __init__(self, shape, dtype, name, reduce: str = "mean"):
        self.local_master_store_rref = _get_local_master_store_rref()
        assert (
            self.local_master_store_rref is not None
        ), "Distributed module is not initialized. Please call sinfer.distributed.init_master_store."
        self.master_store_rref_dict = _get_master_store_rref_dict()
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.reduce = reduce
        self.part_info = get_part_info()
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.master_store_list = []
        for i in range(self.world_size):
            self.master_store_list.append(_get_master_store_name(i))
        self.__init_data()
        self.futs = []

    def __init_data(self):
        rpc_sync(
            self.local_master_store_rref.owner(),
            _call_method,
            args=(
                MasterStore.init_data,
                self.local_master_store_rref,
                self.name,
                self.shape,
                self.dtype,
                self.reduce,
            ),
        )
        rpc.api._barrier(self.master_store_list)
        logging.info(
            f"Init data name={self.name}, shape={self.shape}, dtype={self.dtype}"
        )

    def scatter(self, local_nodes: torch.Tensor, emb: torch.Tensor):
        """
        将mirror节点的embedding发送给master节点.
        注意: 该方法是异步执行的, 不会等待远程worker的scatter方法执行完就直接返回

        Parameters
        ----------
        local_nodes: 计算得到embedding的本地节点ID, shape=(n, )
        emb: local_nodes对应的embedding, shape=(n, hidden_dim)
        """
        # 1. 筛选出边界点:
        # [2, 3, 1, 0, 5]->[2, 3, 0]
        index, local_boundary_nodes = self.part_info.get_boundary_nodes(local_nodes)
        boundary_emb = emb[index]
        # local to global id
        boundary_nodes = self.part_info.local_to_global_id(local_boundary_nodes)
        dst_rank = boundary_nodes % self.world_size
        indices = [
            torch.nonzero(dst_rank == i, as_tuple=True)[0]
            for i in range(self.world_size)
        ]
        if self.reduce == "mean":
            local_degree = self.part_info.get_local_degree(local_boundary_nodes)
            global_degree = self.part_info.get_global_degree(local_boundary_nodes)
            weights = local_degree * 1.0 / global_degree
            boundary_emb = boundary_emb * weights.view(-1, 1).float()
        # 2. scatter: mirror nodes -> master nodes
        for i in range(self.world_size):
            dst_name = _get_master_store_name(i)
            self.futs.append(
                rpc_async(
                    dst_name,
                    _call_method,
                    args=(
                        MasterStore.scatter,
                        self.master_store_rref_dict[dst_name],
                        self.name,
                        boundary_nodes[indices[i]],
                        boundary_emb[indices[i]],
                        self.reduce,
                    ),
                )
            )
        # TODO: 3. num_mirror_nodes对应nodes减1，如果等于0，则表示可以执行apply

    def scatter_all(self, emb: torch.Tensor):
        nodes = torch.arange(emb.shape[0], dtype=torch.int64)
        self.scatter(nodes, emb)
        self.sync()

    def apply_all(self, emb, nonlinear_func=None):
        """
        应用非线性激活函数, 对于内顶点: 直接更新输入`emb`; 对于边界点, 更新`master_store`
        """
        if nonlinear_func is None:
            return
        master_name = _get_master_store_name(MasterStore.rank)
        # 1. apply master nodes
        fut = rpc_async(
            master_name,
            _call_method,
            args=(
                MasterStore.apply_all,
                self.master_store_rref_dict[master_name],
                self.name,
                nonlinear_func,
            ),
        )
        # 2. apply inner nodes
        inner_nodes = self.part_info.get_all_inner_nodes()
        emb[inner_nodes] = nonlinear_func(emb[inner_nodes])
        fut.wait()

    def sync(self):
        for fut in self.futs:
            fut.wait()
        self.futs.clear()
        rpc.api._barrier(self.master_store_list)
        logging.info(f"sync completed")

    def gather_all(self, emb: torch.Tensor):
        """
        收集所有边界点的最新embedding并更新输入`emb`
        """
        local_boundary_nodes = self.part_info.get_all_boundary_nodes()
        # local to global id
        boundary_nodes = self.part_info.local_to_global_id(local_boundary_nodes)
        dst_rank = boundary_nodes % self.world_size
        indices = [
            torch.nonzero(dst_rank == i, as_tuple=True)[0]
            for i in range(self.world_size)
        ]
        futs = []
        # 1. gather: master nodes -> mirror nodes
        for i in range(self.world_size):
            dst_name = _get_master_store_name(i)
            futs.append(
                rpc_async(
                    dst_name,
                    _call_method,
                    args=(
                        MasterStore.gather,
                        self.master_store_rref_dict[dst_name],
                        self.name,
                        boundary_nodes[indices[i]],
                    ),
                )
            )
        # 2. 更新边界点的embedding
        for i, fut in enumerate(futs):
            x = fut.wait()
            emb[local_boundary_nodes[indices[i]]] = x

    def __del__(self):
        if LOCAL_MASTER_STORE_RREF is not None:
            rpc.api._barrier(self.master_store_list)
            master_name = _get_master_store_name(MasterStore.rank)
            rpc_async(
                master_name,
                _call_method,
                args=(
                    MasterStore.delete_data,
                    self.master_store_rref_dict[master_name],
                    self.name,
                ),
            )
            logging.info(f"gas store delete data={self.name} completed")


def alltoall(
    recv_tensors: Union[List[torch.Tensor], torch.Tensor],
    send_tensors: Union[List[torch.Tensor], torch.Tensor],
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if isinstance(send_tensors, torch.Tensor):
        assert len(send_tensors) % world_size == 0
        send_tensors = send_tensors.chunk(world_size)
    if isinstance(recv_tensors, torch.Tensor):
        assert len(recv_tensors) % world_size == 0
        recv_tensors = recv_tensors.chunk(world_size)
    assert len(send_tensors) == world_size
    assert len(recv_tensors) == world_size
    recv_tensors[rank] = send_tensors[rank]
    send_handles = []
    recv_handles = []
    for i in range(1, world_size):
        send_to_rank = (rank + i) % world_size
        recv_from_rank = (rank + world_size - i) % world_size
        send_handles.append(dist.isend(send_tensors[send_to_rank], send_to_rank))
        recv_handles.append(dist.irecv(recv_tensors[recv_from_rank], recv_from_rank))
    for i in range(world_size - 1):
        send_handles[i].wait()
    for i in range(world_size - 1):
        recv_handles[i].wait()


def alltoallv(send_tensors: Union[List[torch.Tensor], torch.Tensor]):
    world_size = dist.get_world_size()
    if isinstance(send_tensors, torch.Tensor):
        assert len(send_tensors) % world_size == 0
        send_tensors = send_tensors.chunk(world_size)
    assert len(send_tensors) == world_size

    dtype = send_tensors[0].dtype
    device = send_tensors[0].device
    send_shape = [
        torch.LongTensor(list(tensor.shape)).to(device) for tensor in send_tensors
    ]
    recv_shape = copy.deepcopy(send_shape)
    alltoall(recv_shape, send_shape)

    recv_tensors = [
        torch.empty(shape.tolist(), dtype=dtype, device=device) for shape in recv_shape
    ]
    alltoall(recv_tensors, send_tensors)
    return recv_tensors
