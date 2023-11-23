from typing import List
import torch
from torch import Tensor
import os


import sinfer.cpp_core as cpp_core


class FeatureStore:
    def __init__(
        self,
        file_path: str,
        offsets: List[int],
        num: int,
        dim: int,
        prefetch=True,
        dtype=torch.float32,
        num_writer_workers: int = 2,
        writer_seq: bool = True,
        dma: bool = False
    ) -> None:
        self.store = cpp_core.FeatureStore(
            file_path,
            offsets,
            num,
            dim,
            prefetch,
            dtype,
            num_writer_workers,
            writer_seq,
            dma
        )
        self.offsets = offsets
        self.num = num
        self.dma = dma

    def gather(self, nodes: Tensor):
        return self.store.gather(nodes)

    def gather_all(self):
        return self.store.gather_all()

    def update_cache(self, part_id: int):
        self.store.update_cache(part_id)

    def write_data(self, nodes: Tensor, data: Tensor):
        self.store.write_data(nodes, data)

    def flush(self):
        self.store.flush()

    def clear_cache(self):
        self.store.clear_cache()

class EmbeddingStore(FeatureStore):
    def __init__(
        self,
        file_path: str,
        offsets: List[int],
        num: int,
        dim: int,
        prefetch=True,
        dtype=torch.float32,
        num_writer_workers: int = 2,
        writer_seq: bool = True,
        dma: bool = False
    ) -> None:
        if not os.path.exists(file_path):
            open(file_path, "w").close()
        self.file_path = file_path
        super().__init__(
            file_path,
            offsets,
            num,
            dim,
            prefetch,
            dtype,
            num_writer_workers,
            writer_seq,
            dma
        )

    def __del__(self):
        # 删除embedding文件
        os.remove(self.file_path)
