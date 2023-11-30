import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import SAGEConv
import time

from sinfer.store import FeatureStore, EmbeddingStore
from sinfer.cpp_core import tensor_free


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers
        self.out_shape = [hidden_channels]
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.out_shape.append(hidden_channels)
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.out_shape.append(out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[: size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x.log_softmax(dim=-1)

    def inference(self, x_all: FeatureStore, loader, device):
        # pbar = tqdm(total=x_all.num * self.num_layers)
        # pbar.set_description("Evaluating")
        import psutil

        process = psutil.Process()
        mem = process.memory_info().rss / (1024 * 1024 * 1024)
        print("before infer mem: {:.4f} GB".format(mem))
        num_nodes = x_all.num
        offsets = x_all.offsets
        dma = x_all.dma
        for i in range(self.num_layers):
            sample_time, gather_time, transfer_time, infer_time, flush_time = (
                0,
                0,
                0,
                0,
                0,
            )
            # 创建一个用于存储embedding的store
            filename = "./embedding-{}.bin".format(i)
            emb_store = EmbeddingStore(
                filename, offsets=offsets, num=num_nodes, dim=self.out_shape[i], dma=dma
            )
            t1 = time.time()
            for step, (batch_size, seeds, n_id, adjs) in enumerate(loader):
                sample_time += time.time() - t1

                t2 = time.time()
                part_id = get_part_id(offsets, seeds[0])
                # 1. 更新缓存: 加载对应partition的embedding进入内存
                x_all.update_cache(part_id)
                # 2. gather: 从SSD和内存去拉取特征数据
                x = x_all.gather(n_id)
                gather_time += time.time() - t2

                t3 = time.time()
                # 3. transfer
                x_cuda = x.to(device)
                if dma:
                    tensor_free(x)
                x_target = x_cuda[:batch_size]
                edge_index, _, _ = adjs[0].to(device)
                torch.cuda.synchronize()
                transfer_time += time.time() - t3

                # 4. 执行推理
                t4 = time.time()
                x_cuda = self.convs[i]((x_cuda, x_target), edge_index)
                if i != self.num_layers - 1:
                    x_cuda = F.relu(x_cuda)
                torch.cuda.synchronize()
                infer_time += time.time() - t4

                x_cpu = x_cuda.to(torch.device("cpu"))
                # 5. 将结果写入到磁盘
                emb_store.write_data(seeds, x_cpu)
                # pbar.update(batch_size)
                t1 = time.time()
                if step % 100 == 0:
                    mem = process.memory_info().rss / (1024 * 1024 * 1024)
                    print(
                        "step: {}, mem: {:.4f} GB, sample time: {:.4f}, gather time: {:.4f}, transfer time: {:.4f}, infer time: {:.4f}".format(
                            step,
                            mem,
                            sample_time,
                            gather_time,
                            transfer_time,
                            infer_time,
                        )
                    )
            t5 = time.time()
            # 同步: 等到所有数据写入磁盘
            emb_store.flush()
            flush_time += time.time() - t5
            # 清除缓存
            x_all.clear_cache()
            x_all = emb_store
            mem = process.memory_info().rss / (1024 * 1024 * 1024)
            print(
                "layer: {},  mem: {:.4f} GB, sample time: {:.4f}, gather time: {:.4f}, transfer time: {:.4f}, infer time: {:.4f}, flush time: {:.4f}".format(
                    i,
                    mem,
                    sample_time,
                    gather_time,
                    transfer_time,
                    infer_time,
                    flush_time,
                )
            )
        # pbar.close()
        return x_all


def get_part_id(offset, n):
    for i in range(len(offset) - 1):
        if n >= offset[i] and n < offset[i + 1]:
            return i
    return len(offset) - 2
