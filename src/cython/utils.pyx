# distutils: language = c++

import numpy as np
cimport numpy as np
from cython.parallel import prange
cimport openmp
from libc.stdlib cimport rand, srand
from libc.time cimport time
cimport cython
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


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


@cython.boundscheck(False)
@cython.wraparound(False)
def rand_assign_partition_nodes(np.ndarray[np.int_t, ndim=2] nodes_to_parts):
    """
    nodes_to_parts[i][j]不为0表示顶点i被划分到了分区j中, 但一个顶点i可能被划分到了多个分区中, 所以
    该方法用于为每个顶点随机选择一个分区
    """
    ## Need to seed random numbers with time otherwise will always get same results
    srand(time(NULL))
    cdef int num_rows = nodes_to_parts.shape[0]
    cdef int num_cols = nodes_to_parts.shape[1]
    cdef np.ndarray[np.int_t, ndim=1] node_to_part_id_map = -np.ones((num_rows, ), dtype=np.int64)
    cdef int i, j
    # 拷贝numpy数组, 避免nogil模式下无法访问numpy数组
    cdef long [:, :] nodes_to_parts_view = nodes_to_parts
    cdef vector[vector[int]] parts = vector[vector[int]](num_rows, vector[int]())
    cdef int part_id
    with nogil:
        for i in prange(num_rows):
            for j in xrange(num_cols):
                if nodes_to_parts_view[i][j] != 0:
                    parts[i].push_back(j)
            if parts[i].size() == 0:
                part_id = rand() % num_cols
            else:
                part_id = parts[i][rand() % parts[i].size()]
            node_to_part_id_map[i] = part_id
    return node_to_part_id_map
