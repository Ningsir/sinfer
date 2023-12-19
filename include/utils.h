#pragma once
#include <ATen/ATen.h>
#include <Python.h>
#include <omp.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <vector>

#define ALIGNMENT 4096

/**
 * 经过dne图划分后，顶点i可能在多个分区中，利用此方法将顶点i随机分配到其中的一个分区中
 * @param nodes_to_parts nodes_to_parts[i][j]不为0表示顶点i被划分到了分区j中,
 但一个顶点i可能被划分到了多个分区中, 所以 该方法用于为每个顶点随机选择一个分区
 * @return nodes_to_part_id_map[i] 表示顶点i被划分到的分区ID
*/
torch::Tensor rand_assign_partition_nodes(const torch::Tensor &nodes_to_parts);

/**
 * pread最多只能读取 0x7ffff000
 (2,147,479,552)字节的数据，如果需要读取大于0x7ffff000
 的数据量，需要多次调用pread
 * Notes: On Linux, read() (and similar system calls) will transfer at most
       0x7ffff000 (2,147,479,552) bytes, returning the number of bytes
       actually transferred.  (This is true on both 32-bit and 64-bit
       systems.)
*/
int64_t pread_wrapper(int fd, void *buf, int64_t count, int64_t offset);

/**
 * On Linux, write() (and similar system calls) will transfer at
       most 0x7ffff000 (2,147,479,552) bytes, returning the number of
       bytes actually transferred.  (This is true on both 32-bit and
       64-bit systems.)
*/
int64_t pwrite_wrapper(int fd, const void *buf, int64_t count, int64_t offset);
