# Reaches around 0.7870 ± 0.0036 test accuracy.

import os
import argparse
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

argparser = argparse.ArgumentParser()
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument("--batch-size", type=int, default=1000)
argparser.add_argument("--num-workers", type=int, default=8)
argparser.add_argument("--num-hiddens", type=int, default=256)
argparser.add_argument("--num-layers", type=int, default=2)
argparser.add_argument(
    "--dataset",
    type=str,
    default="ogbn-papers100M",
    help="dataset name: ogbn-products, ogbn-papers100M",
)
argparser.add_argument("--data-path", type=str, default="/workspace/ningxin/data/")
# train arguments
argparser.add_argument("--train", action="store_true")
argparser.add_argument("--fan-outs", type=str, default="15,10")
argparser.add_argument("--epoch", type=int, default=20)
args = argparser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PygNodePropPredDataset(args.dataset, args.data_path)
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name=args.dataset)
data = dataset[0].to(device, "x", "y")
model_path = os.path.join(
    os.path.dirname(__file__), "{}-sage{}.pt".format(args.dataset, args.num_layers)
)

train_loader = NeighborLoader(
    data,
    input_nodes=split_idx["train"],
    num_neighbors=[int(size) for size in args.fan_outs.split(",")],
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    persistent_workers=True,
)
subgraph_loader = NeighborLoader(
    data,
    input_nodes=None,
    num_neighbors=[-1],
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    persistent_workers=True,
)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all):
        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        import psutil

        process = psutil.Process()
        mem = process.memory_info().rss / (1024 * 1024 * 1024)
        print("before infer mem: {:.4f} GB".format(mem))
        for i in range(self.num_layers):
            xs = []
            sample_time, gather_time, transfer_time, infer_time = 0, 0, 0, 0
            mem = 0
            t1 = time.time()
            for step, batch in enumerate(subgraph_loader):
                sample_time += time.time() - t1
                mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))

                t2 = time.time()
                x = x_all[batch.n_id]
                gather_time += time.time() - t2

                t3 = time.time()
                x = x.to(device)
                edge_index = batch.edge_index.to(device)
                torch.cuda.synchronize()
                transfer_time += time.time() - t3

                t4 = time.time()
                x = self.convs[i](x, edge_index)
                torch.cuda.synchronize()
                infer_time += time.time() - t4

                x = x[: batch.batch_size]
                if i != self.num_layers - 1:
                    x = x.relu()
                xs.append(x.cpu())
                t1 = time.time()
                mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))
            print(
                "layer: {}, peak rss mem: {:.4f} GB, sample time: {:.4f}, gather time: {:.4f}, transfer time: {:.4f}, infer time: {:.4f}".format(
                    i, mem, sample_time, gather_time, transfer_time, infer_time
                )
            )
            x_all = torch.cat(xs, dim=0)
        return x_all


model = SAGE(
    dataset.num_features,
    args.num_hiddens,
    dataset.num_classes,
    num_layers=args.num_layers,
)
model = model.to(device)


def train(epoch):
    model.train()

    pbar = tqdm(total=split_idx["train"].size(0))
    pbar.set_description(f"Epoch {epoch:02d}")

    total_loss = total_correct = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index.to(device))[: batch.batch_size]
        y = batch.y[: batch.batch_size].squeeze().to(torch.int64)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y).sum())
        pbar.update(batch.batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / split_idx["train"].size(0)

    return loss, approx_acc


@torch.no_grad()
def test():
    model.eval()

    out = model.inference(data.x)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["train"]],
            "y_pred": y_pred[split_idx["train"]],
        }
    )["acc"]
    val_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["valid"]],
            "y_pred": y_pred[split_idx["valid"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["test"]],
            "y_pred": y_pred[split_idx["test"]],
        }
    )["acc"]

    return train_acc, val_acc, test_acc


if args.train:
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    for epoch in range(1, 21):
        loss, acc = train(epoch)
        print(f"Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}")
    train_acc, val_acc, test_acc = test()
    print(f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, " f"Test: {test_acc:.4f}")
    torch.save(model.state_dict(), model_path)
else:
    model.load_state_dict(torch.load(model_path))
    start = time.time()
    train_acc, val_acc, test_acc = test()
    print(
        f"infer time: {time.time() - start}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, "
        f"Test: {test_acc:.4f}"
    )
