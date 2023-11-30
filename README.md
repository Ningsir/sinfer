# sinfer: 基于SSD的逐层全图推理

## 方案
处理流程：

1. 预处理阶段：使用metis对图进行切分，然后执行顶点ID重映射，将同一个分区内的顶点ID映射为连续的ID；

2. 推理阶段：将完整的图拓扑结构置于CPU内存，按照从0到N的顺序每次加载一个分区特征到内存，比如将分区i加载到内存，然后推理分区i中的顶点
    - 2.1 将分区i内的特征加载到内存中；
    - 2.2 推理分区i内的顶点：首先执行全采样，然后拉取子图的特征，如果特征不在内存中，则到SSD中去拉取（使用ogbn_products数据集测试发现大约1/20的数据需要到磁盘拉取）；执行推理并将结果写入到SSD中；
    - 2.3 分区i中的顶点推理完成后，执行`i++`回到2.1步骤加载下一个分区的数据进入内存。

逐层推理代码流程：
```python
def inference(self, x_all):
    num_nodes = x_all.shape[0]
    for i in range(self.num_layers):
        # 创建一个用于存储embedding的store
        filename = "./embedding-{}.bin".format(i)
        emb_store = EmbeddingStore(filename, (num_nodes, self.out_shape[i]))
        for part_id, batch_size, n_id, adj, batch in subgraph_loader:
            # 1. 更新缓存: 加载`part_id`分区到内存
            x_all.update_cache(part_id)
            # 2. SSD和缓存中拉取特征
            x = x_all.gather(n_id).to(device)
            x_target = x[: size[1]]
            # 3. 推理
            x = self.convs[i]((x, x_target), adj)
            if i != self.num_layers - 1:
                x = F.relu(x)
            x_cpu = x.to(torch.device("cpu"))
            # 4. 将推理结果写入到SSD
            emb_store.write_data(batch, x_cpu)
        # 同步: 等到所有数据写入磁盘
        emb_store.sync()
        x_all = emb_store
    return x_all
```


## 实验


### 采样后外顶点和内顶点数据占比

采样层数越多，外顶点占比越高；存在重复的顶点导致拉取特征时造成冗余数据访问。

**针对层数越多，外顶点比值越大的问题，可以采样逐层推理的方法，这样就会减少对磁盘的访问**

ogbn_products数据集：

* 一层
```
total time: 1.0694692134857178, inner nodes: 20972631, out nodes: 882567
```

* 两层
```
total time: 4.736915826797485, inner nodes: 255301306, out nodes: 122095568
```

* 三层
```
total time: 40.85113883018494, inner nodes: 350430008, out nodes: 786417293
```

### 性能实验
sinfer、mmap、mem pyg性能测试：
1. 模型层数：2、3
2. 内存限制：4g、8g

**实验结果**

限制使用4G内存时，实验结果如下：

| 模型 | 方法 | 推理时间（s）|
|-----|-------|-------|
|GraphSAGE两层| mmap | 175.83 |
|GraphSAGE两层| sinfer | 95.17 |
|GraphSAGE两层| mmap | 324.20 |
|GraphSAGE两层| sinfer | 141.6 |
> pyg在全内存环境下，限制为4或者8G内存时OOM
> 
> pyg在16G内存: 2层：68.3318 S；3层：93.0755 S


## 环境安装

**安装依赖包**：
```bash
# torch
$ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# dgl
$ pip install  dgl-cu113==0.9.1 -f https://data.dgl.ai/wheels/repo.html
# pyg
$ pip install torch_geometric
# torch_scatter, torch_sparse, torch_cluster
$ pip install torch_scatter torch_sparse torch_cluster
# ogb
$ pip install ogb
# spdlog
$ conda install -c conda-forge spdlog
```
**编译源代码并安装**
```
python setup.py install
```

为了方便调试，也可以不执行安装，使用`--inplace`选项将编译好的库放在`sinfer/`目录下：
```
python setup.py build_ext --inplace
```

设置spdlog的日志级别:
```bash
# warn, err
$ SPDLOG_LEVEL=info python main.py
```

docker 启动:
```shell
docker run --gpus 1 -dit --shm-size="5g" --network=host --name sinfer --memory 16G -v /home/ningxin/:/home sinfer /bin/bash
```
