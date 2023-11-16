#include <stdlib.h>
#include <aio.h>
#include <omp.h>
#include <unistd.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <Python.h>
#include <torch/script.h>
#include <errno.h>
#include <cstring>
#include <inttypes.h>
#include <ATen/ATen.h>
#include <pthread.h>
#include <thread>

#include "gather.h"

#define ALIGNMENT 4096

torch::Tensor gather_mmap(torch::Tensor features, torch::Tensor idx, int64_t feature_dim)
{

    // open file
    int64_t feature_size = feature_dim * sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();
    float *result_buffer = (float *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

    auto features_data = features.data_ptr<float>();
    auto idx_data = idx.data_ptr<int64_t>();

#pragma omp parallel for num_threads(atoi(getenv("GINEX_NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++)
    {
        int64_t i;
        int64_t offset;

        i = idx_data[n];
        memcpy(result_buffer + feature_dim * n, features_data + i * feature_dim, feature_size);
    }

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto result = torch::from_blob(result_buffer, {num_idx, feature_dim}, options);

    return result;
}

torch::Tensor gather_sinfer(std::string feature_file, torch::Tensor idx, int64_t feature_dim, torch::Tensor cache, int64_t cache_start, int64_t cache_end)
{

    int num_threads = atoi(getenv("SINFER_NUM_THREADS"));
    // open file
    // TODO: 不使用O_DIRECT选项是否会提高IO性能
    int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);

    int64_t feature_size = feature_dim * sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();

    float *read_buffer = (float *)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * num_threads);
    float *result_buffer = (float *)aligned_alloc(ALIGNMENT, feature_size * num_idx);

    auto idx_data = idx.data_ptr<int64_t>();
    auto cache_data = cache.data_ptr<float>();
    // auto cache_table_data = cache_table.data_ptr<int32_t>();

#pragma omp parallel for num_threads(num_threads)
    for (int64_t n = 0; n < num_idx; n++)
    {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t cache_entry;
        int64_t read_size;

        i = idx_data[n];
        // cache_entry = cache_table_data[i];
        if (i >= cache_start && i < cache_end)
        {
            memcpy(result_buffer + feature_dim * n, cache_data + (i - cache_start) * feature_dim, feature_size);
        }
        else
        {
            offset = i * feature_size;
            aligned_offset = offset & (long)~(ALIGNMENT - 1);
            residual = offset - aligned_offset;

            if (residual + feature_size > ALIGNMENT)
            {
                read_size = ALIGNMENT * 2;
            }
            else
            {
                read_size = ALIGNMENT;
            }

            if (pread(feature_fd, read_buffer + (ALIGNMENT * 2 * omp_get_thread_num()) / sizeof(float), read_size, aligned_offset) == -1)
            {
                fprintf(stderr, "ERROR: %s\n", strerror(errno));
            }
            memcpy(result_buffer + feature_dim * n, read_buffer + (ALIGNMENT * 2 * omp_get_thread_num() + residual) / sizeof(float), feature_size);
        }
    }

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    auto result = torch::from_blob(result_buffer, {num_idx, feature_dim}, options);

    free(read_buffer);
    close(feature_fd);

    return result;
}

torch::Tensor gather_mem(std::string feature_file, int64_t start, int64_t end, int64_t feature_dim)
{
    // open file
    int feature_fd = open(feature_file.c_str(), O_RDONLY);

    int64_t feature_size = feature_dim * sizeof(float);

    int64_t num = end - start;

    int64_t total_size = num * feature_size;

    int64_t offset = start * feature_size;
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    torch::Tensor output_tensor = torch::empty({num, feature_dim}, options);

    if (pread(feature_fd, output_tensor.data_ptr(), total_size, offset) == -1)
    {
        fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }
    close(feature_fd);
    return output_tensor;
}

torch::Tensor gather_ssd(std::string feature_file, torch::Tensor idx, int64_t feature_dim)
{
    int num_threads = atoi(getenv("SINFER_NUM_THREADS"));

    // open file
    int feature_fd = open(feature_file.c_str(), O_RDONLY);

    int64_t num_idx = idx.numel();
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    torch::Tensor output_tensor = torch::empty({num_idx, feature_dim}, options);

    int64_t feature_size = feature_dim * sizeof(float);

    auto out_data = output_tensor.data_ptr<float>();
    auto idx_data = idx.data_ptr<int64_t>();

#pragma omp parallel for num_threads(num_threads)
    for (int64_t n = 0; n < num_idx; n++)
    {
        int64_t offset = idx_data[n] * feature_size;
        if (pread(feature_fd, (char *)out_data + n * feature_size, feature_size, offset) == -1)
        {
            fprintf(stderr, "ERROR: %s\n", strerror(errno));
        }
    }
    close(feature_fd);
    return output_tensor;
}

void cache_gather_(torch::Tensor cache_node_data, torch::Tensor cache_idx, torch::Tensor cache, int64_t cache_start, int64_t feature_dim, torch::Tensor output_tensor)
{
    int num_threads = atoi(getenv("SINFER_NUM_THREADS"));
    int64_t feature_size = feature_dim * sizeof(float);

    auto cache_node_ptr = cache_node_data.data_ptr<int64_t>();
    auto cache_idx_ptr = cache_idx.data_ptr<int64_t>();
    auto cache_ptr = cache.data_ptr<float>();
    auto output_ptr = output_tensor.data_ptr<float>();

    int64_t num = cache_node_data.numel();
#pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < num; i++)
    {
        int64_t node = cache_node_ptr[i];
        int64_t cache_offset = (node - cache_start) * feature_size;
        int64_t output_offset = cache_idx_ptr[i] * feature_size;
        memcpy((char *)output_ptr + output_offset, (char *)cache_ptr + cache_offset, feature_size);
    }
}

void ssd_gather_(std::string feature_file, torch::Tensor ssd_node_data, torch::Tensor ssd_idx, int64_t feature_dim, torch::Tensor output_tensor)
{
    int num_threads = atoi(getenv("SINFER_NUM_THREADS"));
    int64_t feature_size = feature_dim * sizeof(float);
    int feature_fd = open(feature_file.c_str(), O_RDONLY);

    auto ssd_node_ptr = ssd_node_data.data_ptr<int64_t>();
    auto ssd_idx_ptr = ssd_idx.data_ptr<int64_t>();
    auto output_ptr = output_tensor.data_ptr<float>();

    int64_t num = ssd_node_data.numel();
#pragma omp parallel for num_threads(num_threads)
    for (int64_t i = 0; i < num; i++)
    {
        int64_t ssd_offset = ssd_node_ptr[i] * feature_size;
        int64_t output_offset = ssd_idx_ptr[i] * feature_size;
        if (pread(feature_fd, (char *)output_ptr + output_offset, feature_size, ssd_offset) == -1)
        {
            fprintf(stderr, "ERROR: %s\n", strerror(errno));
        }
    }
    close(feature_fd);
}

/*将缓存和内存中的数据分开执行gather*/
torch::Tensor gather_sinfer1(std::string feature_file, torch::Tensor idx, int64_t feature_dim, torch::Tensor cache, int64_t cache_start, int64_t cache_end)
{
    int64_t num_idx = idx.numel();

    torch::Tensor mask1 = (idx >= cache_start) & (idx <= cache_end);
    // 在缓存中拉取特征的顶点
    torch::Tensor cache_node_data = torch::masked_select(idx, mask1);
    // 在缓存中拉取特征的顶点在idx中对应的索引
    torch::Tensor cache_idx = torch::nonzero(mask1).squeeze();

    torch::Tensor mask2 = (idx < cache_start) | (idx > cache_end);
    torch::Tensor ssd_node_data = torch::masked_select(idx, mask2);
    torch::Tensor ssd_idx = torch::nonzero(mask2).squeeze();

    assert(idx.numel() == cache_node_data.numel() + ssd_node_data.numel());

    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .layout(torch::kStrided)
                       .device(torch::kCPU)
                       .requires_grad(false);
    torch::Tensor output = torch::empty({num_idx, feature_dim}, options);

    // cache_gather_(cache_node_data, cache_idx, cache, cache_start, feature_dim, output);
    // ssd_gather_(feature_file, ssd_node_data, ssd_idx, feature_dim, output);
    std::thread t1(cache_gather_, cache_node_data, cache_idx, cache, cache_start, feature_dim, output);
    std::thread t2(ssd_gather_, feature_file, ssd_node_data, ssd_idx, feature_dim, output);

    t1.join();
    t2.join();
    return output;
}
