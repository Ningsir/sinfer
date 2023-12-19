import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import argparse
import os
import time


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        for _ in range(num_layers - 2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, num_workers):
        """Conduct layer-wise inference to get all the node embeddings."""
        import psutil

        process = psutil.Process()
        mem = process.memory_info().rss / (1024 * 1024 * 1024)
        print("before infer memory usage: %.4fG" % mem)
        feat = g.ndata["feat"]
        del g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()),
            sampler,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            sample_time, gather_time, copy_time, infer_time = 0, 0, 0, 0
            mem = 0
            t1 = time.time()
            start = time.time()
            with dataloader.enable_cpu_affinity():
                for _, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                    mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))
                    sample_time += time.time() - t1
                    t2 = time.time()
                    x = feat[input_nodes]
                    gather_time += time.time() - t2

                    t3 = time.time()
                    x = x.to(device)
                    blocks = [block.to(device) for block in blocks]
                    torch.cuda.synchronize()
                    copy_time += time.time() - t3

                    t4 = time.time()
                    h = layer(blocks[0], x)  # len(blocks) = 1
                    if l != len(self.layers) - 1:
                        h = F.relu(h)
                        h = self.dropout(h)
                    # by design, our output nodes are contiguous
                    y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
                    torch.cuda.synchronize()
                    infer_time += time.time() - t4
                    t1 = time.time()
                    mem = max(mem, process.memory_info().rss / (1024 * 1024 * 1024))
            print(
                "layer: {}, peak rss mem: {:.4f} GB, time: {:.4f}, sample time: {:.4f}, gather time: {:.4f}, transfer time: {:.4f}, infer time: {:.4f}".format(
                    l,
                    mem,
                    time.time() - start,
                    sample_time,
                    gather_time,
                    copy_time,
                    infer_time,
                )
            )
            feat = y
        return y


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return compute_acc(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(device, graph, nid, model, batch_size, num_workers):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size, num_workers
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return compute_acc(pred, label)


def train(args, device, g, dataset, model):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [
            int(size) for size in args.fan_outs.split(",")
        ],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(args.epoch):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"].to(torch.int64)
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-hiddens", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--data-path", type=str, default="/workspace/ningxin/data/")
    # train arguments
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--fan-outs", type=str, default="5,10,15")
    parser.add_argument("--epoch", type=int, default=20)
    args = parser.parse_args()
    print(args)
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"{args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(
        DglNodePropPredDataset(args.dataset, root=args.data_path)
    )
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, args.num_hiddens, out_size, args.num_layers).to(device)
    model_path = os.path.join(
        os.path.dirname(__file__), f"{args.dataset}-sage{args.num_layers}.pt"
    )
    if args.train:
        # model training
        print("Training...")
        train(args, device, g, dataset, model)
        torch.save(model.state_dict(), model_path)
        model.eval()
        # test the model
        print("Testing...")
        acc = layerwise_infer(
            device,
            g,
            dataset.test_idx,
            model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print("Test Accuracy {:.4f}".format(acc.item()))
    else:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # test the model
        print("Testing...")
        start = time.time()
        acc = layerwise_infer(
            device,
            g,
            dataset.test_idx,
            model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(
            "infer time: {:.4f}, Test Accuracy {:.4f}".format(
                time.time() - start, acc.item()
            )
        )
