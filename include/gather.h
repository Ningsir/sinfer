#include <torch/extension.h>

torch::Tensor gather_sinfer(std::string feature_file, torch::Tensor idx, int64_t feature_dim, torch::Tensor cache, int64_t cache_start, int64_t cache_end);
torch::Tensor gather_mem(std::string feature_file, int64_t start, int64_t end, int64_t feature_dim);
torch::Tensor gather_ssd(std::string feature_file, torch::Tensor idx, int64_t feature_dim);
torch::Tensor gather_sinfer1(std::string feature_file, torch::Tensor idx, int64_t feature_dim, torch::Tensor cache, int64_t cache_start, int64_t cache_end);