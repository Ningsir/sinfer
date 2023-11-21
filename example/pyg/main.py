import argparse
import time
import os
import sys

import torch as th


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
from sinfer.dataloader import SinferPygDataloader
from sinfer.data import SinferDataset
from sinfer.store import FeatureStore

from sage import SAGE


def acc(out, labels, train_idx, val_idx, test_idx):
    from ogb.nodeproppred import Evaluator

    evaluator = Evaluator(name="ogbn-products")
    y_true = labels.reshape(-1, 1)
    y_pred = out.argmax(dim=-1, keepdim=True)
    train_acc = evaluator.eval(
        {
            "y_true": y_true[train_idx],
            "y_pred": y_pred[train_idx],
        }
    )["acc"]
    val_acc = evaluator.eval(
        {
            "y_true": y_true[val_idx],
            "y_pred": y_pred[val_idx],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": y_true[test_idx],
            "y_pred": y_pred[test_idx],
        }
    )["acc"]
    return train_acc, val_acc, test_acc


if __name__ == "__main__":
    default_model_path = os.path.join(os.path.dirname(__file__), "sage.pt")
    argparser = argparse.ArgumentParser("layer based inference")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--n-classes", type=int, default=47)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=2)
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--model-path", type=str, default=default_model_path)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--save-pred", type=str, default="")
    argparser.add_argument("--wd", type=float, default=0)
    args = argparser.parse_args()
    if args.gpu >= 0:
        device = th.device("cuda:%d" % args.gpu)
    else:
        device = th.device("cpu")
    os.environ["SINFER_NUM_THREADS"] = "16"
    data = SinferDataset("/home/ningxin/data/ogbn-products-ssd-infer-part16")
    print(data)
    # run(args, data)
    nodes = th.arange(0, data.num_nodes, dtype=th.int64)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    dataloader = SinferPygDataloader(data.indptr, data.indices, [-1], nodes, **kwargs)

    model = SAGE(data.feat_dim, args.num_hidden, data.num_classes, args.num_layers).to(
        device
    )
    model.load_state_dict(th.load(args.model_path))
    for i in range(10):
        # TODO: bugfix: 一个FeatureStore被重复使用会导致系统死锁
        feat_store = FeatureStore(
            data.feat_path, data.offsets, data.num_nodes, data.feat_dim
        )
        model.eval()
        start = time.time()
        with th.no_grad():
            out = model.inference(feat_store, dataloader, device)
        print("infer time: {} s".format(time.time() - start))
        out = out.gather_all()
        labels = data.lables
        train_idx, val_idx, test_idx = data.train_idx, data.val_idx, data.test_idx
        train_acc, val_acc, test_acc = acc(out, labels, train_idx, val_idx, test_idx)
        print(
            "train acc: {}, val acc: {}, test acc: {}".format(
                train_acc, val_acc, test_acc
            )
        )
