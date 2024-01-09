
# 基于PyG实现的分布式全图推理

1. 数据准备：处理完成后会将原始数据下载到`/home/data/ogbn_products`目录，处理后的数据在`/home/data/dist/ogbn-products`目录下
    ```bash
    python process_data.py --dataset ogbn-products --data-path /home/data --part-path /home/data/ogbn-products-dne-part8/ --num-parts 8 --output /home/data/dist --output-num-parts 2
    ```
    目录结构如下：
    ```
    ogbn-products/
    ├── part-0
    │   ├── conf.json
    │   ├── coo.bin
    │   ├── feat.bin
    │   ├── global_degree.bin
    │   ├── indices.bin
    │   ├── indptr.bin
    │   ├── labels.bin
    │   ├── local_degree.bin
    │   ├── origin_nodes.bin
    │   ├── test_idx.bin
    │   ├── train_idx.bin
    │   └── val_idx.bin
    └── part-1
        ├── conf.json
        ├── coo.bin
        ├── feat.bin
        ├── global_degree.bin
        ├── indices.bin
        ├── indptr.bin
        ├── labels.bin
        ├── local_degree.bin
        ├── origin_nodes.bin
        ├── test_idx.bin
        ├── train_idx.bin
        └── val_idx.bin
    ```
2. 配置分布式环境：配置ssh免密登录
3. 在两台机器上执行推理：
    
    3.1 启动node1：
    ```bash
    python -m torch.distributed.launch --nproc_per_node=1  --nnodes=2 --node_rank=0 --master_addr="172.17.0.5" --master_port=1234 main.py --dataset ogbn-products --data-path /home/data/dist/
    ```
    3.2 启动node2：
    ```bash
    python -m torch.distributed.launch --nproc_per_node=1  --nnodes=2 --node_rank=1 --master_addr="172.17.0.5" --master_port=1234 main.py --dataset ogbn-products --data-path /home/data/dist/
    ```