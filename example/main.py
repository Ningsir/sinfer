import argparse
import time
import os
import sys

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from sinfer.dataloader import DataLoader
from sinfer.data import SinferDataset

from sage import SAGE

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, test_nid, device, batch_size, num_workers=0):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device, batch_size, num_workers)
    model.train()
    return (
        compute_acc(pred[val_nid], labels[val_nid]),
        compute_acc(pred[test_nid], labels[test_nid]),
        pred,
    )


def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels

    
#### Entry point
def run(args, data):
    coo_row, coo_col = data.coo()
    graph = dgl.graph((coo_row, coo_col))
    print(graph)
    kwargs = {'batch_size': args.batch_size,
        'drop_last': False,
        }
    infer_dataloader = DataLoader(graph, data.feat_path, data.feat_dim, data.offsets, prefetch=True, **kwargs)
    # Define model and optimizer
    model = SAGE(
        data.feat_dim,
        args.num_hidden,
        data.num_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    data_time = 0
    infer_time = 0
    model.eval()
    start = time.time()
    with th.no_grad():
        t1 = time.time()
        #TODO: 从磁盘中获取特征太慢
        for step, (input_nodes, seeds, blocks) in enumerate(infer_dataloader):
            data_time += time.time() - t1
            t2 = time.time()
            batch_inputs = blocks[0].srcdata['feat']
            batch_pred = model(blocks, batch_inputs)
            infer_time += time.time() - t2
            if step % 100 == 0:
                print("Infer step: {}, data time: {}, infer time: {}".format(step, data_time, infer_time))
            t1 = time.time()

    infer_dataloader.shutdown()
    print("Infer time: {}, data time: {}, infer time: {}".format(time.time() - start, data_time, infer_time))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-parts", type=int, default=8)
    argparser.add_argument("--num-buffers", type=int, default=4)
    argparser.add_argument("--prefetch", type=bool, default=True)
    argparser.add_argument("--n-classes", type=int, default=47)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=1)
    # 5, 10, 15
    argparser.add_argument("--fan-out", type=str, default="25,10")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--val-batch-size", type=int, default=10000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--save-pred", type=str, default="")
    argparser.add_argument("--wd", type=float, default=0)
    args = argparser.parse_args()
    # if args.gpu >= 0:
    #     device = th.device("cuda:%d" % args.gpu)
    # else:
    #     device = th.device("cpu")
    
    data = SinferDataset("/home/data/ogbn-products-ssd-infer")
    print(data)
    run(args, data)
