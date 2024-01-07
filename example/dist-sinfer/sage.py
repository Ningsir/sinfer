import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import time

import sinfer.distributed as dist


class SAGE(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, aggr="mean"
    ):
        super(SAGE, self).__init__()
        self.aggr = aggr
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

    @torch.no_grad()
    def inference(
        self, x_all: torch.Tensor, loader, device, num_total_nodes, pipeline=True
    ):
        import psutil

        process = psutil.Process()
        mem = process.memory_info().rss / (1024 * 1024 * 1024)
        print("before infer mem: {:.4f} GB".format(mem))

        for i in range(self.num_layers):
            sample_time, gather_time, transfer_time, infer_time, flush_time = (
                0,
                0,
                0,
                0,
                0,
            )
            mem = 0
            x_output = []
            gas_store = dist.GASStore(
                (num_total_nodes // dist.get_world_size() + 1, self.out_shape[i]),
                dtype=torch.float32,
                name=f"gas_store_{i}",
                reduce=self.aggr,
            )
            t1 = time.time()
            for step, (batch_size, seeds, n_id, adjs) in enumerate(loader):
                sample_time += time.time() - t1
                mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))

                t2 = time.time()
                x = x_all[n_id]
                edge_index, _, _ = adjs[0]
                gather_time += time.time() - t2

                t3 = time.time()
                # 3. transfer
                if x.device != device:
                    x = x.to(device)
                    edge_index = edge_index.to(device)
                    torch.cuda.synchronize()
                transfer_time += time.time() - t3

                # 4. 执行推理
                t4 = time.time()
                x_target = x[:batch_size]
                x = self.convs[i]((x, x_target), edge_index)
                x_cpu = x.cpu()
                if pipeline:
                    gas_store.scatter(seeds, x_cpu)
                torch.cuda.synchronize()
                infer_time += time.time() - t4

                x_output.append(x_cpu)
                t1 = time.time()
                mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))
            x_all = torch.cat(x_output, dim=0)
            t5 = time.time()
            if not pipeline:
                gas_store.scatter_all(x_all)
            gas_store.sync()
            if i != self.num_layers - 1:
                nonlinear_func = F.relu
            else:
                nonlinear_func = None
            gas_store.apply_all(x_all, nonlinear_func=nonlinear_func)
            gas_store.gather_all(x_all)
            flush_time += time.time() - t5
            mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))
            print(
                "layer: {},  peak rss mem: {:.4f} GB, sample time: {:.4f}, gather time: {:.4f}, transfer time: {:.4f}, infer time: {:.4f}, apply+gather time: {:.4f}".format(
                    i,
                    mem,
                    sample_time,
                    gather_time,
                    transfer_time,
                    infer_time,
                    flush_time,
                )
            )
        return x_all
