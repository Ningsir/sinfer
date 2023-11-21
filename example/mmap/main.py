from typing import List, Optional, Callable, NamedTuple, Tuple
import os
import json
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor

from sage import SAGE


def sample_adj_mmap(
    rowptr, col, subset: torch.Tensor, num_neighbors: int, replace: bool = False
):
    rowptr, col, n_id, e_id = torch.ops.torch_sparse.sample_adj(
        rowptr, col, subset, num_neighbors, replace
    )
    out = SparseTensor(
        rowptr=rowptr,
        row=None,
        col=col,
        value=None,
        sparse_sizes=(subset.size(0), n_id.size(0)),
        is_sorted=True,
    )
    return out, n_id


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class MMAPDataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        indptr,
        indices,
        sizes: List[int],
        node_idx: Tensor,
        num_nodes: Optional[int] = None,
        transform: Callable = None,
        **kwargs
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        if "dataset" in kwargs:
            del kwargs["dataset"]

        self.indptr = indptr
        self.indices = indices
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.transform = transform

        if node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super(MMAPDataLoader, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs
        )

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = sample_adj_mmap(self.indptr, self.indices, n_id, size, False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]

            adjs.append(Adj(adj_t, e_id, size))

        adjs = adjs[::-1]
        out = (batch_size, n_id, adjs, batch)
        out = self.transform(*out) if self.transform is not None else out
        return out

    def __repr__(self):
        return "{}(sizes={})".format(self.__class__.__name__, self.sizes)


def get_data(path):
    indptr_path = os.path.join(path, "indptr.bin")
    indices_path = os.path.join(path, "indices.bin")
    features_path = os.path.join(path, "features.bin")
    labels_path = os.path.join(path, "labels.bin")
    conf_path = os.path.join(path, "conf.json")
    split_idx_path = os.path.join(path, "split_idx.pth")

    conf = json.load(open(conf_path, "r"))

    indptr = np.fromfile(indptr_path, dtype=conf["indptr_dtype"]).reshape(
        tuple(conf["indptr_shape"])
    )
    indices = np.memmap(
        indices_path,
        mode="r",
        shape=tuple(conf["indices_shape"]),
        dtype=conf["indices_dtype"],
    )
    features_shape = conf["features_shape"]
    features = np.memmap(
        features_path,
        mode="r",
        shape=tuple(features_shape),
        dtype=conf["features_dtype"],
    )
    labels = np.fromfile(
        labels_path, dtype=conf["labels_dtype"], count=conf["num_nodes"]
    ).reshape(tuple([conf["labels_shape"][0]]))

    indptr = torch.from_numpy(indptr)
    indices = torch.from_numpy(indices)
    features = torch.from_numpy(features)
    labels = torch.from_numpy(labels)

    num_nodes = conf["num_nodes"]
    feat_dim = conf["features_shape"][1]
    num_classes = conf["num_classes"]

    split_idx = torch.load(split_idx_path)
    train_idx = split_idx["train"]
    val_idx = split_idx["valid"]
    test_idx = split_idx["test"]
    return (
        indptr,
        indices,
        features,
        labels,
        feat_dim,
        num_nodes,
        num_classes,
        split_idx,
    )


@torch.no_grad()
def full_inference():
    import psutil

    process = psutil.Process()
    mem = process.memory_info().rss / (1024 * 1024 * 1024)
    print("before infer mem: {} GB".format(mem))
    model.eval()
    sample_time, gather_time, copy_time, infer_time = 0, 0, 0, 0
    start = time.time()
    t1 = time.time()
    # Sample
    for step, (batch_size, ids, adjs, batch) in enumerate(full_infer_loader):
        sample_time += time.time() - t1
        t2 = time.time()
        # Gather
        batch_inputs = features[ids]
        gather_time += time.time() - t2
        t3 = time.time()
        # Transfer
        batch_inputs_cuda = batch_inputs.to(device)
        adjs = [adj.to(device) for adj in adjs]
        torch.cuda.synchronize()
        copy_time += time.time() - t3
        t4 = time.time()
        # Forward
        out = model(batch_inputs_cuda, adjs)
        torch.cuda.synchronize()
        infer_time += time.time() - t4
        if step % 100 == 0:
            mem = process.memory_info().rss / (1024 * 1024 * 1024)
            print("infer mem: {} GB".format(mem))
            print(
                "Infer step: {}, adj size: {}, sample_time: {}, gather time: {}, infer time: {}".format(
                    step, adjs[0].size, sample_time, gather_time, infer_time
                )
            )
        t1 = time.time()
    print("Infer time: {}".format(time.time() - start))


def train(epoch, train_loader):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    model.train()
    x_all = features
    for i in range(epoch):
        total_loss = total_correct = 0
        for step, (batch_size, n_id, adjs, batch) in enumerate(train_loader):
            x = x_all[n_id].to(device)
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()
            out = model(x, adjs)[:batch_size]

            y = labels[batch].squeeze().to(device)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-1).eq(y).sum())

        loss = total_loss / len(train_loader)
        approx_acc = total_correct / split_idx["train"].size(0)
        print("Epoch: {}, loss: {}. train acc: {}".format(i, loss, approx_acc))


def acc(out, labels, split_idx):
    from ogb.nodeproppred import Evaluator

    evaluator = Evaluator(name="ogbn-products")
    y_true = labels.reshape(-1, 1)
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gpu", type=int, default=0)
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--num-workers", type=int, default=8)
    argparser.add_argument("--num-hiddens", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=2)
    argparser.add_argument(
        "--data_path", type=str, default="/home/ningxin/data/ogbn-products-mmap"
    )
    # train arguments
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--fan-outs", type=str, default="15, 10")
    argparser.add_argument("--epoch", type=int, default=20)
    args = argparser.parse_args()

    (
        indptr,
        indices,
        features,
        labels,
        feat_dim,
        num_nodes,
        num_classes,
        split_idx,
    ) = get_data(args.data_path)
    labels = labels.long()
    device = torch.device("cuda:%d" % args.gpu)
    torch.cuda.set_device(device)
    model = SAGE(feat_dim, args.num_hiddens, num_classes, num_layers=args.num_layers)
    model = model.to(device)
    model_path = os.path.join(os.path.dirname(__file__), "sage.pt")
    if not args.train:
        model.load_state_dict(torch.load(model_path))
        all_nodes = torch.arange(0, num_nodes, dtype=torch.int64)
        full_infer_loader = MMAPDataLoader(
            indptr,
            indices,
            node_idx=all_nodes,
            sizes=[-1],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        # full_inference()
        model.eval()
        start = time.time()
        out = model.inference(features, full_infer_loader, device)
        train_acc, val_acc, test_acc = acc(out, labels, split_idx)
        print(
            "infer time: {}, train acc: {}, val acc: {}, test acc: {}".format(
                time.time() - start, train_acc, val_acc, test_acc
            )
        )
    else:
        train_loader = MMAPDataLoader(
            indptr,
            indices,
            node_idx=split_idx["train"],
            sizes=[int(size) for size in args.fan_outs.split(",")],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        train(args.epoch, train_loader)
        torch.save(model.state_dict(), model_path)
