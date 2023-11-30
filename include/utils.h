#pragma once

#include <ATen/ATen.h>
#include <Python.h>
#include <omp.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <vector>

/**
 * 经过dne图划分后，顶点i可能在多个分区中，利用此方法将顶点i随机分配到其中的一个分区中
 * @param nodes_to_parts nodes_to_parts[i][j]不为0表示顶点i被划分到了分区j中,
 但一个顶点i可能被划分到了多个分区中, 所以 该方法用于为每个顶点随机选择一个分区
 * @return nodes_to_part_id_map[i] 表示顶点i被划分到的分区ID
*/
torch::Tensor rand_assign_partition_nodes(const torch::Tensor& nodes_to_parts) {
  srand(NULL);
  auto sizes = nodes_to_parts.sizes();
  int64_t row = sizes[0], col = sizes[1];
  torch::Tensor nodes_to_part_id_map = torch::ones({row}, torch::kInt64);
  auto map_ptr = nodes_to_part_id_map.data_ptr<int64_t>();
#pragma omp parallel for
  for (int64_t i = 0; i < row; i++) {
    std::vector<int64_t> part_ids;
    part_ids.reserve(col);
    for (int64_t j = 0; j < col; j++) {
      if (nodes_to_parts[i][j].item<int64_t>() != 0) {
        part_ids.push_back(j);
      }
    }
    int64_t id;
    if (part_ids.size() == 0) {
      id = rand() % col;
    } else {
      id = part_ids[rand() % part_ids.size()];
    }
    map_ptr[i] = id;
  }
  return nodes_to_part_id_map;
}
