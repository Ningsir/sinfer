#pragma once

#include <torch/extension.h>

void tensor_free(torch::Tensor t);
