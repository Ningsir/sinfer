

1. 使用metis对图进行切分
2. 对图顶点重新编号，在同一个分区内的顶点ID连续
3. 内存保存全图数据，每次从磁盘中加载一个分区顶点和对应特征进入内存
4. 利用UVA机制执行采样，然后提取特征，存放在磁盘中的数据去磁盘中拉取

```cpp

class MemFeature{
public:
    void swap(){

    }

    torch::Tensor gather(torch::Tensor nodes){

    }

private:
};

class DiskFeature{

public:
    torch::Tensor gather(torch::Tensor nodes){

    }

private:

};
```

使用cython加速：
```
python setup.py build_ext --inplace
```