# distutils: language = c++

import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport openmp

def coo_to_adj_list(np.ndarray[np.int_t, ndim=1] row,
                    np.ndarray[np.int_t, ndim=1] col,):
    cdef int num_rows = np.amax(row) + 1
    cdef int num_cols = np.amax(col) + 1
    cdef dict adj_list = {i: [] for i in range(num_rows)}
    cdef int i

    # 利用OpenMP并行化循环
    with nogil:
        for i in prange(row.shape[0]):
            # 使用OpenMP原子操作以避免竞争条件
            with gil:
                adj_list[row[i]].append(col[i])
    return adj_list


def coo_to_adj_list_1(np.ndarray[np.int_t, ndim=1] row,
                    np.ndarray[np.int_t, ndim=1] col):
    cdef int num_rows = np.amax(row) + 1
    cdef int num_cols = np.amax(col) + 1
    cdef dict adj_list = {i: [] for i in range(num_rows)}
    cdef int i

    for i in range(row.shape[0]):
        adj_list[row[i]].append(col[i])

    return adj_list