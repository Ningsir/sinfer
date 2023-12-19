from typing import List
import torch
from torch import Tensor
import os
import numpy as np


import sinfer.cpp_core as cpp_core


def torch2numpy(torch_dtype):
    if torch_dtype == torch.float16:
        return np.float16
    elif torch_dtype == torch.float32:
        return np.float32
    elif torch_dtype == torch.float64:
        return np.float64
    elif torch_dtype == torch.uint8:
        return np.uint8
    elif torch_dtype == torch.int8:
        return np.int8
    elif torch_dtype == torch.int16:
        return np.int16
    elif torch_dtype == torch.int32:
        return np.int32
    elif torch_dtype == torch.int64:
        return np.int64
    else:
        raise ValueError("Unsupported data type: {}".format(torch_dtype))


class Store(object):
    def __init__(self, num: int, dim: int, dtype=torch.float32):
        self.num = num
        self.dim = dim
        self.dtype = dtype

    def gather(self, nodes: Tensor) -> Tensor:
        raise NotImplementedError("gather not implemented")

    def gather_all(self) -> Tensor:
        raise NotImplementedError("gather_all not implemented")

    def update_cache(self, part_id: int):
        pass

    def write_data(self, nodes: Tensor, data: Tensor):
        raise NotImplementedError("write_data not implemented")

    def flush(self):
        pass

    def clear_cache(self):
        pass


class InMemStore(Store):
    def __init__(self, num: int, dim: int, dtype=torch.float32, file_path: str = None):
        super().__init__(num, dim, dtype)
        if file_path is not None:
            data = np.fromfile(file_path, dtype=torch2numpy(dtype))
            self.__data = torch.from_numpy(data.reshape(num, dim))
        else:
            self.__data = torch.empty(num, dim, dtype=dtype)

    def gather(self, nodes: Tensor):
        return self.__data[nodes]

    def gather_all(self):
        return self.__data

    def write_data(self, nodes: Tensor, data: Tensor):
        self.__data[nodes] = data


class SSDStore(Store):
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
        dma: bool = False,
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
            dma,
        )
        self.offsets = offsets
        self.dma = dma
        self.file_path = file_path
        super().__init__(num, dim, dtype)

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


class SSDEmbeddingStore(SSDStore):
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
        dma: bool = False,
    ) -> None:
        if not os.path.exists(file_path):
            open(file_path, "w").close()
        # self.file_path = file_path
        super().__init__(
            file_path,
            offsets,
            num,
            dim,
            prefetch,
            dtype,
            num_writer_workers,
            writer_seq,
            dma,
        )

    def __del__(self):
        # 删除embedding文件
        # os.remove(self.file_path)
        pass


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
        dma: bool = False,
        in_mem: bool = False,
    ) -> None:
        if not in_mem:
            self.store = cpp_core.FeatureStore(
                file_path,
                offsets,
                num,
                dim,
                prefetch,
                dtype,
                num_writer_workers,
                writer_seq,
                dma,
            )
        else:
            self.store = InMemStore(num, dim, dtype, file_path)
        self.num = num
        self.offsets = offsets
        self.dma = dma
        self.file_path = file_path

    def gather(self, nodes: Tensor):
        return self.store.gather(nodes)

    def gather_all(self):
        return self.store.gather_all()

    def update_cache(self, part_id: int):
        self.store.update_cache(part_id)

    def write_data(self, nodes: Tensor, data: Tensor):
        raise RuntimeError("write_data is not supported for FeatureStore")

    def flush(self):
        raise RuntimeError("flush is not supported for FeatureStore")

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
        dma: bool = False,
        in_mem: bool = False,
    ) -> None:
        if not in_mem:
            if not os.path.exists(file_path):
                open(file_path, "w").close()
            # self.file_path = file_path
        else:
            file_path = None
        super().__init__(
            file_path,
            offsets,
            num,
            dim,
            prefetch,
            dtype,
            num_writer_workers,
            writer_seq,
            dma,
            in_mem,
        )

    def write_data(self, nodes: Tensor, data: Tensor):
        self.store.write_data(nodes, data)

    def flush(self):
        self.store.flush()

    def __del__(self):
        # 删除embedding文件
        if self.file_path is not None:
            os.remove(self.file_path)
