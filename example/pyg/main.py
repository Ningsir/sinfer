import argparse
import time
import os
import sys

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_sparse import SparseTensor


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
from sinfer.dataloader import PygDataLoader
from sinfer.data import SinferDataset
from sinfer.cpp_core import tensor_free

from sage import SAGE


#### Entry point
def run(args, data):
    indptr, indices = data.indptr, data.indices
    kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    # offsets = [0, data.num_nodes]
    infer_dataloader = PygDataLoader(
        indptr,
        indices,
        data.feat_path,
        data.feat_dim,
        data.offsets,
        prefetch=True,
        **kwargs
    )
    # Define model and optimizer
    model = SAGE(data.feat_dim, args.num_hidden, data.num_classes, args.num_layers).to(
        device
    )
    # loss_fcn = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    data_time = 0
    copy_time = 0
    infer_time = 0
    model.eval()
    start = time.time()
    with th.no_grad():
        t1 = time.time()
        for step, (batch_size, seeds, adjs, feat) in enumerate(infer_dataloader):
            data_time += time.time() - t1
            t2 = time.time()
            feat = feat.to(device)
            adjs = [adj.to(device) for adj in adjs]
            th.cuda.synchronize()
            copy_time += time.time() - t2
            t3 = time.time()
            batch_pred = model(feat, adjs)
            # tensor_free(batch_inputs)
            th.cuda.synchronize()
            infer_time += time.time() - t3
            if step % 100 == 0:
                print(
                    "Infer step: {}, adj size: {}, data time: {}, copy time: {}, infer time: {}".format(
                        step, adjs[0].size, data_time, copy_time, infer_time
                    )
                )
            t1 = time.time()

    infer_dataloader.shutdown()
    print(
        "Infer time: {}, data time: {}, copy time: {},  infer time: {}".format(
            time.time() - start, data_time, copy_time, infer_time
        )
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("layer based inference")
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
        default=0,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--save-pred", type=str, default="")
    argparser.add_argument("--wd", type=float, default=0)
    args = argparser.parse_args()
    # if args.gpu >= 0:
    #     device = th.device("cuda:%d" % args.gpu)
    # else:
    #     device = th.device("cpu")
    device = th.device("cuda:%d" % args.gpu)
    # device = th.device("cpu")
    os.environ["SINFER_NUM_THREADS"] = "16"
    data = SinferDataset("/home/data/ogbn-products-ssd-infer")
    print(data)
    run(args, data)
