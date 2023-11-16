from typing import Optional, Tuple, NamedTuple
import torch as th
import dgl
import queue
import time
from dgl.dataloading import MultiLayerFullNeighborSampler
from torch_sparse import SparseTensor

from sinfer.cpp_core import gather_mem, gather_sinfer, gather_sinfer1


class StopEvent:
    def __init__(self):
        pass


# 读取下一轮的特征数据进入内存
def _read_next_feature(args_queue, data_queue):
    while True:
        try:
            r = args_queue.get()
        except queue.Empty:
            continue
        if isinstance(r, StopEvent):
            break
        data_path, start, end, feat_dim = r
        print("start:{}, end:{}".format(start, end))
        data = gather_mem(data_path, start, end, feat_dim)
        print("read finished")
        data_queue.put((data, start, end))


class SSDInferSampler(MultiLayerFullNeighborSampler):
    def __init__(
        self, num_layers, feature_dim, disk_data_path, cache_feat, cache_range, **kwargs
    ):
        super().__init__(num_layers, **kwargs)
        self.disk_data_path = disk_data_path
        self.cache_feat = cache_feat
        self.cache_range = cache_range
        self.feature_dim = feature_dim
        self.sample_time, self.gather_time, self.copy_time = 0, 0, 0
        self.sample_nodes = 0

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        t1 = time.time()
        seed_nodes, output_nodes, blocks = super().sample_blocks(
            g, seed_nodes, exclude_eids
        )
        self.sample_time += time.time() - t1

        t2 = time.time()
        seed_nodes = seed_nodes.to(th.device("cpu"))
        block_device = blocks[0].device
        # 从CPU缓存和磁盘中读取特征数据
        feat = gather_sinfer1(
            self.disk_data_path,
            seed_nodes,
            self.feature_dim,
            self.cache_feat,
            self.cache_range[0],
            self.cache_range[1],
        )
        blocks[0].srcdata["feat"] = feat.to(block_device)
        self.gather_time += time.time() - t2
        self.sample_nodes += seed_nodes.shape[0]

        # t3 = time.time()
        # device = th.device('cuda:0')
        # blocks = [block.int().to(device) for block in blocks]
        # th.cuda.synchronize()
        # self.copy_time += time.time() - t3
        return seed_nodes, output_nodes, blocks

    def __del__(self):
        speed = self.sample_nodes * self.feature_dim * 4 / (self.gather_time * 1e9)
        print(
            "sample time: {}, gather time: {}, gather speed: {} GB/s".format(
                self.sample_time, self.gather_time, speed
            )
        )


# TODO: 更新embedding，以支持逐层推理
class DataLoader(object):
    """
    用于全图推理的DataLoader
    Parameters
    ----------
    offsets : List[int]
        记录推理样本的偏移量
    kwargs :
        传递给`dgl.dataloading.DataLoader`.
    """

    def __init__(self, graph, data_path, feat_dim, offsets, prefetch=True, **kwargs):
        if "shuffle" in kwargs.keys():
            kwargs.pop("shuffle")
        self.graph = graph
        self.data_path = data_path
        self.feat_dim = feat_dim
        self.offsets = offsets
        self.prefetch = prefetch
        self.kwargs = kwargs
        self.exit = False
        self.offsets_list = self.__offsets_list()
        self.len = len(self.offsets_list)
        # 使用spawn进程以支持GPU
        multiprocessing_context = th.multiprocessing.get_context("spawn")
        self.args_queue = multiprocessing_context.Queue()
        self.data_queue = multiprocessing_context.Queue()
        w = multiprocessing_context.Process(
            target=_read_next_feature, args=(self.args_queue, self.data_queue)
        )
        w.daemon = True
        w.start()

    def __offsets_list(self):
        # [(start1, end1), (start2, end2), ..., (start_n, end_n)]
        offsets_list = []
        for i in range(len(self.offsets) - 1):
            start = self.offsets[i]
            end = self.offsets[i + 1]
            offsets_list.append((start, end))
        return offsets_list

    def __iter__(self):
        self.idx = 0
        self.len = len(self.offsets_list)
        # prefetch
        if self.prefetch and self.idx < len(self.offsets_list):
            start, end = self.offsets_list[self.idx]
            self.idx += 1
            self.args_queue.put((self.data_path, start, end, self.feat_dim))
        # 获取dgl的DataLoader
        self.__update_dgl_dataloader()
        return self

    # 加载下一个分区的特征进入内存，更新sampler，然后更新dataloader
    def __update_dgl_dataloader(self):
        if self.idx < len(self.offsets_list):
            start, end = self.offsets_list[self.idx]
            self.idx += 1
            self.args_queue.put((self.data_path, start, end, self.feat_dim))
        # 读取内存特征
        self.mem_feat, start, end = self.data_queue.get()
        # 缓存的顶点ID范围
        self.cache_range = (start, end)
        self.len -= 1
        nodes = th.arange(start, end, dtype=th.int64)
        # 更新sampler
        sampler = SSDInferSampler(
            1, self.feat_dim, self.data_path, self.mem_feat, (start, end)
        )
        self.dgl_dataloader = iter(
            dgl.dataloading.DataLoader(
                self.graph, nodes, sampler, shuffle=False, **self.kwargs
            )
        )

    def __next__(self):
        try:
            value = next(self.dgl_dataloader)
            return value
        except StopIteration:
            if self.len > 0:
                # 更新DGL DataLoader
                self.__update_dgl_dataloader()
                value = next(self.dgl_dataloader)
                return value
            else:
                raise StopIteration

    def shutdown(self):
        if not self.exit:
            self.data_queue.put(StopEvent())
            self.exit = True

    def __del__(self):
        self.shutdown()


def sample_adj(
    rowptr, col, subset: th.Tensor, num_neighbors: int, replace: bool = False
):
    rowptr, col, n_id, e_id = th.ops.torch_sparse.sample_adj(
        rowptr, col, subset, num_neighbors, replace
    )
    out = SparseTensor(
        rowptr=rowptr,
        row=None,
        col=col,
        value=None,
        sparse_sizes=(subset.size(0), n_id.size(0)),
        is_sorted=True,
    )
    return out, n_id


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[th.Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class PygSSDInferSampler:
    def __init__(
        self,
        indptr,
        indices,
        fan_out,
        feature_dim,
        disk_data_path,
        cache_feat,
        cache_range,
    ):
        self.indptr = indptr
        self.indices = indices
        self.fan_out = fan_out
        self.disk_data_path = disk_data_path
        self.cache_feat = cache_feat
        self.cache_range = cache_range
        self.feature_dim = feature_dim
        self.sample_time, self.gather_time, self.copy_time = 0, 0, 0
        self.sample_nodes = 0

    def sample(self, batch):
        if not isinstance(batch, th.Tensor):
            batch = th.tensor(batch)

        batch_size: int = len(batch)
        t1 = time.time()
        adjs = []
        n_id = batch
        for size in self.fan_out:
            adj_t, n_id = sample_adj(self.indptr, self.indices, n_id, size, False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            adjs.append(Adj(adj_t, e_id, size))
        self.sample_time += time.time() - t1
        t2 = time.time()
        adjs = adjs[::-1]
        feat = gather_sinfer1(
            self.disk_data_path,
            n_id,
            self.feature_dim,
            self.cache_feat,
            self.cache_range[0],
            self.cache_range[1],
        )
        out = (batch_size, n_id, adjs, feat)
        self.gather_time += time.time() - t2
        self.sample_nodes += n_id.shape[0]
        # out = self.transform(*out) if self.transform is not None else out
        return out

    def __del__(self):
        speed = self.sample_nodes * self.feature_dim * 4 / (self.gather_time * 1e9)
        print(
            "sample time: {}, gather time: {}, gather speed: {} GB/s".format(
                self.sample_time, self.gather_time, speed
            )
        )


class PygDataLoader(object):
    """
    用于全图推理的DataLoader
    Parameters
    ----------
    offsets : List[int]
        记录推理样本的偏移量
    kwargs :
        传递给`torch.utils.data.DataLoader`.
    """

    def __init__(
        self, indptr, indices, data_path, feat_dim, offsets, prefetch=True, **kwargs
    ):
        if "shuffle" in kwargs.keys():
            kwargs.pop("shuffle")
        self.indptr = indptr
        self.indices = indices
        self.data_path = data_path
        self.feat_dim = feat_dim
        self.offsets = offsets
        self.prefetch = prefetch
        self.kwargs = kwargs
        self.exit = False
        self.offsets_list = self.__offsets_list()
        self.len = len(self.offsets_list)
        # 使用spawn进程以支持GPU
        multiprocessing_context = th.multiprocessing.get_context("spawn")
        self.args_queue = multiprocessing_context.Queue()
        self.data_queue = multiprocessing_context.Queue()
        w = multiprocessing_context.Process(
            target=_read_next_feature, args=(self.args_queue, self.data_queue)
        )
        w.daemon = True
        w.start()

    def __offsets_list(self):
        # [(start1, end1), (start2, end2), ..., (start_n, end_n)]
        offsets_list = []
        for i in range(len(self.offsets) - 1):
            start = self.offsets[i]
            end = self.offsets[i + 1]
            offsets_list.append((start, end))
        return offsets_list

    def __iter__(self):
        self.idx = 0
        self.len = len(self.offsets_list)
        # prefetch
        if self.prefetch and self.idx < len(self.offsets_list):
            start, end = self.offsets_list[self.idx]
            self.idx += 1
            self.args_queue.put((self.data_path, start, end, self.feat_dim))
        self.__update_pyg_dataloader()
        return self

    # 加载下一个分区的特征进入内存，更新sampler，然后更新dataloader
    def __update_pyg_dataloader(self):
        if self.idx < len(self.offsets_list):
            start, end = self.offsets_list[self.idx]
            self.idx += 1
            self.args_queue.put((self.data_path, start, end, self.feat_dim))
        # 读取内存特征
        self.mem_feat, start, end = self.data_queue.get()
        # 缓存的顶点ID范围
        self.cache_range = (start, end)
        self.len -= 1
        nodes = th.arange(start, end, dtype=th.int64)
        # 更新sampler
        self.sampler = PygSSDInferSampler(
            self.indptr,
            self.indices,
            [-1],
            self.feat_dim,
            self.data_path,
            self.mem_feat,
            (start, end),
        )
        self.pyg_dataloader_iter = iter(
            th.utils.data.DataLoader(
                nodes, collate_fn=self.sampler.sample, shuffle=False, **self.kwargs
            )
        )

    def __next__(self):
        try:
            value = next(self.pyg_dataloader_iter)
            return value
        except StopIteration:
            if self.len > 0:
                # 更新DGL DataLoader
                self.__update_pyg_dataloader()
                value = next(self.pyg_dataloader_iter)
                return value
            else:
                raise StopIteration

    def shutdown(self):
        if not self.exit:
            self.data_queue.put(StopEvent())
            self.exit = True

    def __del__(self):
        self.shutdown()
