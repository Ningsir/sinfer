#include <stdlib.h>
#include <aio.h>
#include <omp.h>
#include <unistd.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <errno.h>
#include <cstring>
#include <inttypes.h>
#include <ATen/ATen.h>
#include <pthread.h>
#define ALIGNMENT 4096

torch::Tensor gather_mmap(torch::Tensor features, torch::Tensor idx, int64_t feature_dim){

    // open file
    int64_t feature_size = feature_dim*sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();
    float* result_buffer = (float*)aligned_alloc(ALIGNMENT, feature_size*num_idx);

    auto features_data = features.data_ptr<float>();
    auto idx_data = idx.data_ptr<int64_t>();

    #pragma omp parallel for num_threads(atoi(getenv("GINEX_NUM_THREADS")))
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
 
        i = idx_data[n];
        memcpy(result_buffer+feature_dim*n, features_data+i*feature_dim, feature_size);
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    auto result = torch::from_blob(result_buffer, {num_idx, feature_dim}, options);

    return result;

}

torch::Tensor gather_sinfer(std::string feature_file, torch::Tensor idx, int64_t feature_dim, torch::Tensor cache, int64_t cache_start, int64_t cache_end){

    const int num_threads = 16;
    // open file
    int feature_fd = open(feature_file.c_str(), O_RDONLY | O_DIRECT);

    int64_t feature_size = feature_dim*sizeof(float);
    int64_t read_size = feature_size;

    int64_t num_idx = idx.numel();

    float* read_buffer = (float*)aligned_alloc(ALIGNMENT, ALIGNMENT * 2 * num_threads);
    float* result_buffer = (float*)aligned_alloc(ALIGNMENT, feature_size * num_idx);

    auto idx_data = idx.data_ptr<int64_t>();
    auto cache_data = cache.data_ptr<float>();
    // auto cache_table_data = cache_table.data_ptr<int32_t>();

    #pragma omp parallel for num_threads(num_threads)
    for (int64_t n = 0; n < num_idx; n++) {
        int64_t i;
        int64_t offset;
        int64_t aligned_offset;
        int64_t residual;
        int64_t cache_entry;
        int64_t read_size;

        i = idx_data[n];
        // cache_entry = cache_table_data[i];
        if (i >= cache_start && i < cache_end) {
            memcpy(result_buffer+feature_dim*n, cache_data+(i - cache_start) * feature_dim, feature_size);
        }
        else {
            offset = i * feature_size;
            aligned_offset = offset&(long)~(ALIGNMENT-1);
            residual = offset - aligned_offset;

            if (residual+feature_size > ALIGNMENT){
                read_size = ALIGNMENT * 2;
            }
            else {
                read_size = ALIGNMENT;
            }

            if (pread(feature_fd, read_buffer+(ALIGNMENT*2*omp_get_thread_num())/sizeof(float), read_size, aligned_offset) == -1){
                fprintf(stderr, "ERROR: %s\n", strerror(errno));
            }
            memcpy(result_buffer+feature_dim*n, read_buffer+(ALIGNMENT*2*omp_get_thread_num()+residual)/sizeof(float), feature_size);
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

torch::Tensor gather_mem(std::string feature_file, int64_t start, int64_t end, int64_t feature_dim){
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

    if (pread(feature_fd, output_tensor.data_ptr(), total_size, offset) == -1){
        fprintf(stderr, "ERROR: %s\n", strerror(errno));
    }
    close(feature_fd);
    return output_tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gather_sinfer", &gather_sinfer, "gather for sinfer", py::call_guard<py::gil_scoped_release>());
    m.def("gather_mem", &gather_mem, "gather", py::call_guard<py::gil_scoped_release>());
    m.def("gather_mmap", &gather_mmap, "gather for PyG+", py::call_guard<py::gil_scoped_release>());
}
