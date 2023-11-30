#!/bin/bash
num_layers=(2 3)
mem=4g
data_time=$(date +"%Y-%m-%d")
# ssd或者nvm
device=ssd
base_dir=/workspace/ningxin/data
# base_dir=/home/data

for i in {1..5}; do
    for layer in "${num_layers[@]}"
    do
        # python example/mem/main.py --num-layers ${layer} --data-path ${base_dir} >> example/mem/log/${data_time}-${device}-sage${layer}-${mem}.log
        
        # mmap
        python example/mmap/main.py --num-layers ${layer} --data-path ${base_dir}/ogbn-products-mmap >> example/mmap/log/${data_time}-${device}-sage${layer}-${mem}.log

        # sinfer
        python example/pyg/main.py --num-layers ${layer} --data-path ${base_dir}/ogbn-products-ssd-infer-part16 >> example/pyg/log/${data_time}-${device}-sage${layer}-${mem}.log

        # sinfer with dma
        # python example/pyg/main.py --dma --num-layers ${layer} --data-path ${base_dir}/ogbn-products-ssd-infer-part16 >> example/pyg/log/${data_time}-${device}-sage${layer}-dma-${mem}.log

        # sinfer, dne partition
        python example/pyg/main.py --num-layers ${layer} --data-path ${base_dir}/ogbn-products-ssd-infer-dne-part8 >> example/pyg/log/${data_time}-${device}-dne-sage${layer}-${mem}.log
    done
done
