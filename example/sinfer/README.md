# 基于pyg实现的SSD推理

1. 数据准备：处理完成后会将原始数据下载到`/home/data/ogbn_products`目录，处理后的数据在`/home/data/ogbn-products-ssd-infer-part8`目录下
    ```bash
    python prepare_dataset.py --dataset ogbn-products --data-path /home/data --num-parts 8 --part-method metis
    ```
    目录结构如下：
    ```
    ogbn-products-ssd-infer-part8
    ├── conf.json
    ├── coo.bin
    ├── feat.bin
    ├── indices.bin
    ├── indptr.bin
    ├── labels.bin
    ├── offsets.txt
    ├── test_idx.bin
    ├── train_idx.bin
    └── val_idx.bin
    ```
2. 执行训练：训练完会将模型保存到当前目录的`sage2.pt`文件中
    ```bash
    python train.py --data-path /home/data
    ```
3. 执行推理
    ```bash
    python main.py --data-path /home/data/ogbn-products-ssd-infer-part8 --num-layers 2
    ```

4. 一致性测试：执行全局推理时，使用`--runs`选项设置推理的轮数，最后会比较计算得到的embedding结果是否一致。
    ```bash
    python main.py --data-path /home/data/ogbn-products-ssd-infer-part8 --num-layers 2 --runs 10
    ```
