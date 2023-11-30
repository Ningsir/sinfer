# 基于内存映射的全图推理


1. 数据准备：处理完成后会将原始数据下载到`/home/data/ogbn_products`目录，处理后的数据在`/home/data/ogbn-products-mmap`目录下
    ```bash
    python data_process.py --dataset ogbn-products --data_path /home/data
    ```

    目录结构如下：
    
    ```
    ogbn-products-mmap/
    ├── conf.json
    ├── features.bin
    ├── indices.bin
    ├── indptr.bin
    ├── labels.bin
    └── split_idx.pth
    ```

2. 执行训练：训练完会将模型保存到当前目录的`sage2.pt`文件中
    ```bash
    python train.py --data-path /home/data/ogbn-products-mmap --train --num-layers 2
    ```

3. 执行推理
    ```bash
    python train.py --data-path /home/data/ogbn-products-mmap
    ```
