#pragma once

#include <ATen/ATen.h>
#include <Python.h>
#include <aio.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <unistd.h>

#include <cstring>
#include <thread>

#include "gather.h"

#define ALIGNMENT 4096

/**
 * 在缓存和ssd(利用DMA)中读取数据
 * @param cache 顶点ID在[cache_start, cache_end)范围内的特征缓存在cache中
 * */
torch::Tensor gather_cache_ssd_dma(std::string feature_file,
                                   const torch::Tensor& idx,
                                   int64_t feature_dim,
                                   const torch::Tensor& cache,
                                   int64_t cache_start,
                                   int64_t cache_end);

/**
 * 读取顶点ID在[start, end)范围内的数据
 * */
torch::Tensor gather_range_with_fd(int fd,
                                   int64_t start,
                                   int64_t end,
                                   int64_t feature_dim);

/**
 * 读取顶点ID在[start, end)范围内的数据
 * */
torch::Tensor gather_range(std::string feature_file,
                           int64_t start,
                           int64_t end,
                           int64_t feature_dim);

/**
 * 从磁盘读取数据
 * @param feature_fd 文件描述符
 * */
torch::Tensor gather_ssd_with_fd(int feature_fd,
                                 const torch::Tensor& idx,
                                 int64_t feature_dim);

/**
 * 从磁盘读取数据
 * @param feature_file 数据文件路径
 * */
torch::Tensor gather_ssd(std::string feature_file,
                         const torch::Tensor& idx,
                         int64_t feature_dim);

/**
 * 将缓存和内存中的数据分开执行gather
 * @param cache 顶点ID在[cache_start, cache_end)范围内的特征缓存在cache中
 * */
torch::Tensor gather_cache_ssd_with_fd(int fd,
                                       const torch::Tensor& idx,
                                       int64_t feature_dim,
                                       const torch::Tensor& cache,
                                       int64_t cache_start,
                                       int64_t cache_end);

/**
 * 将缓存和内存中的数据分开执行gather
 * @param cache 顶点ID在[cache_start, cache_end)范围内的特征缓存在cache中
 * */
torch::Tensor gather_cache_ssd(std::string feature_file,
                               const torch::Tensor& idx,
                               int64_t feature_dim,
                               const torch::Tensor& cache,
                               int64_t cache_start,
                               int64_t cache_end);
