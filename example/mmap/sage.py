import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import SAGEConv
import time
import os


class MapTensor:
    def __init__(self, filename, shape, dtype=np.float32):
        self.filename = filename
        self.shape = shape
        self.dtype = dtype
        mmap_array = np.memmap(
            self.filename, dtype=self.dtype, mode="w+", shape=self.shape
        )
        self.mmap_tensor = torch.from_numpy(mmap_array)

    def write_data(self, ids, data):
        self.mmap_tensor[ids] = data[:]

    def tensor(self):
        return self.mmap_tensor

    def __del__(self):
        del self.mmap_tensor  # 删除mmap数组
        import os

        os.remove(self.filename)  # 从磁盘中删除文件


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.out_shape = [hidden_channels]
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.out_shape.append(hidden_channels)
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.out_shape.append(out_channels)
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device, data_path):
        import psutil

        process = psutil.Process()
        mem = process.memory_info().rss / (1024 * 1024 * 1024)
        print("before infer mem: {:.4f} GB".format(mem))
        # pbar = tqdm(total=x_all.size(0) * self.num_layers)
        # pbar.set_description("Evaluating")
        num_nodes = x_all.shape[0]
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        total_edges = 0
        for i in range(self.num_layers):
            # create a mmap tensor
            filename = os.path.join(data_path, "embedding-{}.bin".format(i))
            emb_mmap = MapTensor(filename, (num_nodes, self.out_shape[i]))
            sample_time, gather_time, transfer_time, infer_time, write_time = 0, 0, 0, 0, 0
            mem = 0
            t1 = time.time()
            for step, (batch_size, n_id, adj, batch) in enumerate(subgraph_loader):
                sample_time += time.time() - t1
                mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))

                t2 = time.time()
                # gather
                x = x_all[n_id]
                gather_time += time.time() - t2

                t3 = time.time()
                # transfer
                x = x.to(device)
                edge_index, _, size = adj[0].to(device)
                x_target = x[: size[1]]
                total_edges += edge_index.size(1)
                torch.cuda.synchronize()
                transfer_time += time.time() - t3

                t4 = time.time()
                # infer
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                torch.cuda.synchronize()
                infer_time += time.time() - t4

                t5 = time.time()
                x_cpu = x.to(torch.device("cpu"))
                emb_mmap.write_data(batch, x_cpu)
                write_time += time.time() - t5
                # pbar.update(batch_size)
                t1 = time.time()
                mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))
            mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))
            print(
                "layer: {}, peak rss mem: {:.4f} GB, sample time: {:.4f}, gather time: {:.4f}, transfer time: {:.4f}, infer time: {:.4f}, write time: {:.4f}".format(
                    i, mem, sample_time, gather_time, transfer_time, infer_time, write_time
                )
            )
            x_all = emb_mmap.tensor()

        # pbar.close()

        return x_all
