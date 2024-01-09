# sinfer

sinfer 是一个基于大规模图数据的GNN全图推理系统，在单机环境下，支持基于SSD的GNN逐层全图，同时还支持基于分布式的GNN全图推理。

TODO：
- [ ] 增加实验，完善文档
- [ ] 实现基于UVA的采样和特征提取
- [ ] 分布式：实现异步的分布式全图推理；消息压缩以减小通信量
- [ ] 实现GPU和NVM的直接访问

## 基于SSD的GNN全图推理
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

## 分布式GNN全图推理

首先采用`veterx-cut`的方法对图进行分区，如果一个顶点在多个分区中，则称之为边界点。边界点又分为master节点和mirror节点，所有边界点的最新embedding都会被同步到master节点。

数据处理完成后采用`GAS`的模型执行逐层的全图推理，`GAS`模型主要提供以下编程API：
1. scatter：将mirror节点的embedding同步到master节点；
2. apply：对master节点的embedding应用非线性激活函数的更新，以得到最新的embedding；
3. gather：mirror节点从master节点收集最新的embedding。


基于`GAS`模型的分布式全图推理代码如下所示：
```python
# 初始化GAS以及分布式环境
sinfer.distributed.init_gas_store(rank, world_size, part_info)

# 逐层推理
for layer in range(self.num_layers):
    # 1. 创建GASStore：用于存储边界点的embedding
    gas_store = sinfer.distributed.GASStore((num_total_nodes // world_size + 1, num_hiddens[layer]), dtype=torch.float32, name=f"gas_store_layer_{layer}")
    x_output = []
    # 2. 执行推理
    for batch_size, seeds, n_id, adj in dataloader:
        x = x_all.[n_id].to(device)
        x_target = x[: size[1]]
        # 2.1 推理
        x = convs[layer]((x, x_target), adj)
        x_output.append(x)
        # 2.2 scatter：将mirror节点的embedding同步到master节点
        gas_store.scatter(seeds, x)
    # 3. 同步: 等待所有scatter执行完成
    gas_store.sync()
    x_all = torch.cat(x_output, dim=0)
    # 4. apply：应用非线性激活更新
    gas_store.apply(x_all, nonlinear_func)
    # 5. gather
    gas_store.gather(x_all)

sinfer.distributed.shutdown()
```

## 使用教程

## 实验

**数据集**
> 

| Name                                                         |      #Nodes |       #Edges | #Tasks | Split Type |
| :----------------------------------------------------------- | ----------: | ------------: | -----: | :--------: |
| [ogbn-products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) |   2,449,029 |    61,859,140 |      1 | Sales rank |
| [ogbn-papers100M](https://ogb.stanford.edu/docs/nodeprop/#ogbn-papers100M) | 111,059,956 | 1,615,685,872 |      1 |    Time    |

### 单机测试

**单机实验环境**
* 系统：ubuntu20.04
* 显卡：Nvidia A800
* CPU：Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz（64 core 128 thread）
* SSD：2T
* 内存：1T内存(DDR4 3200)

**测试系统**：

* sinfer: CPU采样+基于SSD的特征拉取，GPU推理
* mem+pyg：CPU采样，GPU推理
* mmap+pyg：基于内存映射的CPU采样，GPU推理
* mem+dgl: CPU采样，GPU推理
> * mem+dgl: UVA采样，GPU推理
> * sinfer: UVA采样+基于SSD的特征拉取，GPU推理

#### 一致性测试

执行全局推理时，使用`--runs`选项设置推理的轮数，最后会比较计算得到的embedding结果是否一致。

```bash
python example/sinfer/main.py --data-path /home/data/ogbn-products-ssd-infer-part8 --num-layers 2 --runs 10
```

#### 不限内存


不限制内存使用（1T内存）,内存占用情况：

* dgl内存占用最多，在products数据集中，是sinfer的3.56-4.98X；papers100中，是sinfer的4.91-9.27X
* distdgl在papers100中没有跑起来，在products数据集中和sinfer内存占用差不多（graphserver端的内存占用没有统计，所以内存占用少）；
* pyg：在products数据集中，是sinfer的2.54-3.33X；papers100中，是sinfer的3.61-6.81X
* mmap：在products数据集中，是sinfer的1.26-2X；papers100中，是sinfer的2.92-5.31X

![image-20231225122156230](https://raw.githubusercontent.com/Ningsir/image/main/image/image-20231225122156230.png)

 

![image-20231225122219891](https://raw.githubusercontent.com/Ningsir/image/main/image/image-20231225122219891.png)

不限制内存：全图推理运行时间
> 注意：PyG在采样时就将特征传输到GPU上，所以PyG的transfer时间相对于其他系统更少.

![image-20231225105011711](https://raw.githubusercontent.com/Ningsir/image/main/image/image-20231225105011711.png)

#### 限制内存

在限制内存的条件下，只测试了mmap和sinfer两个系统，因为其他系统都会OOM

* 在products数据集上，限制使用4G和8G内存，sinfer相对于mmap有1.37-2.53倍的加速比；
* 在papers100M数据集上，限制使用100G内存，sinfer相对于mmap有10倍左右的加速。

![image-20231225112353829](https://raw.githubusercontent.com/Ningsir/image/main/image/image-20231225112353829.png)



![image-20231225112408454](https://raw.githubusercontent.com/Ningsir/image/main/image/image-20231225112408454.png)


### 分布式测试

#### dist-sinfer vs DistDGL

系统：
* DistDGL：分布式DGL
* dist-sinfer：分布式sinfer
* dist-sinfer(pipeline): 分布式sinfer，计算和scatter通信重叠。

结果：dist-sinfer相对于DistDGL有1.85-3.67X的加速；dist-sinfer(pipeline)相对于DistDGL有1.86-6.3X的加速。

![image-20240109112530506](https://raw.githubusercontent.com/Ningsir/image/main/image/image-20240109112530506.png)

#### 采样后外顶点和内顶点数据占比

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
