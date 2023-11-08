import torch as th
import dgl
import queue
import time
from dgl.dataloading import MultiLayerFullNeighborSampler

from sinfer.cpp_core import gather_mem, gather_sinfer

class StopEvent():
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
        print(data)
        data_queue.put((data, start, end))


class SSDInferSampler(MultiLayerFullNeighborSampler):
    def __init__(self, num_layers, feature_dim, disk_data_path, cache_feat, cache_range, **kwargs):
        super().__init__(num_layers, **kwargs)
        self.disk_data_path = disk_data_path
        self.cache_feat = cache_feat
        self.cache_range = cache_range
        self.feature_dim = feature_dim
        
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
       seed_nodes, output_nodes, blocks = super().sample_blocks(g, seed_nodes, exclude_eids)
       # 从CPU缓存和磁盘中读取特征数据
       blocks[0].srcdata['feat'] = gather_sinfer(self.disk_data_path, seed_nodes, self.feature_dim, self.cache_feat, self.cache_range[0], self.cache_range[1])
       return seed_nodes, output_nodes, blocks


#TODO: 更新embedding，以支持逐层推理
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
        multiprocessing_context = th.multiprocessing.get_context('spawn')
        self.args_queue = multiprocessing_context.Queue()
        self.data_queue = multiprocessing_context.Queue()
        w = multiprocessing_context.Process(
                target=_read_next_feature,
                args=(self.args_queue, self.data_queue))
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
        sampler = SSDInferSampler(1, self.feat_dim, self.data_path, self.mem_feat, (start, end))
        self.dgl_dataloader = iter(dgl.dataloading.DataLoader(
            self.graph,
            nodes,
            sampler,
            shuffle=False,
            **self.kwargs
        ))

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
