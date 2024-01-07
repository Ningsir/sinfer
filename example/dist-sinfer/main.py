import argparse
import time
import os
import sys
import torch as th
import torch.multiprocessing as mp

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)
from sinfer.dataloader import SinferPygDataloader
from sinfer.data import SinferDataset
import sinfer.distributed as dist

from sage import SAGE


def acc(out, labels, train_idx, val_idx, test_idx, dataset_name):
    import torch.distributed as thdist
    from ogb.nodeproppred import Evaluator

    thdist.init_process_group(
        backend="gloo", world_size=dist.get_world_size(), rank=dist.get_rank()
    )
    evaluator = Evaluator(name=dataset_name)
    y_true = labels.reshape(-1, 1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    def total_num_nodes(nodes_idx):
        num_idx = th.tensor([nodes_idx.size(0)], dtype=th.int64)
        thdist.all_reduce(num_idx, op=thdist.ReduceOp.SUM)
        return num_idx[0].item()

    num_total_train = total_num_nodes(train_idx)
    num_total_val = total_num_nodes(val_idx)
    num_total_test = total_num_nodes(test_idx)

    def average_acc(acc):
        acc = th.tensor([acc], dtype=th.float32)
        thdist.all_reduce(acc, op=thdist.ReduceOp.SUM)
        return acc[0].item()

    train_acc = evaluator.eval(
        {
            "y_true": y_true[train_idx],
            "y_pred": y_pred[train_idx],
        }
    )["acc"]
    average_train_acc = average_acc(train_acc * train_idx.size(0) / num_total_train)
    val_acc = evaluator.eval(
        {
            "y_true": y_true[val_idx],
            "y_pred": y_pred[val_idx],
        }
    )["acc"]
    average_val_acc = average_acc(val_acc * val_idx.size(0) / num_total_val)
    test_acc = evaluator.eval(
        {
            "y_true": y_true[test_idx],
            "y_pred": y_pred[test_idx],
        }
    )["acc"]
    average_test_acc = average_acc(test_acc * test_idx.size(0) / num_total_test)
    thdist.destroy_process_group()
    return average_train_acc, average_val_acc, average_test_acc


def main(rank, world_size, args):
    if args.gpu >= 0:
        device = th.device("cuda:%d" % args.gpu)
    else:
        device = th.device("cpu")
    print(f"rank: {rank}, device: {device}")
    data_path = os.path.join(args.data_path, args.dataset, f"part-{rank}")
    data = SinferDataset(data_path)
    print(data)
    origin_total_num_nodes = data.origin_total_num_nodes
    print(f"origin total num nodes: {origin_total_num_nodes}")
    origin_nodes, local_degree, global_degree = (
        data.origin_nodes,
        data.local_degree,
        data.global_degree,
    )
    local_nodes = th.arange(origin_nodes.shape[0], dtype=th.int64)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    dataloader = SinferPygDataloader(
        data.indptr, data.indices, [-1], local_nodes, **kwargs
    )
    print(f"num edges: {data.indices.shape[0]}")
    part_info = dist.PartInfo(local_nodes, origin_nodes, local_degree, global_degree)
    model = SAGE(data.feat_dim, args.num_hidden, data.num_classes, args.num_layers).to(
        device
    )
    model_path = os.path.join(
        os.path.dirname(__file__), f"{args.dataset}-sage{args.num_layers}.pt"
    )
    model.load_state_dict(th.load(model_path, map_location=device))
    all_embs = []
    for i in range(args.runs):
        dist.init_master_store(rank, world_size, part_info)
        model.eval()
        start = time.time()
        with th.no_grad():
            out = model.inference(
                data.feat,
                dataloader,
                device,
                origin_total_num_nodes,
                pipeline=args.pipeline,
            )
        print("infer time: {:.4f}".format(time.time() - start))
        all_embs.append(out)
        labels = data.labels
        train_idx, val_idx, test_idx = data.train_idx, data.val_idx, data.test_idx
        train_acc, val_acc, test_acc = acc(
            out, labels, train_idx, val_idx, test_idx, args.dataset
        )
        print(
            "rank: {}, train acc: {:.4f}, val acc: {:.4f}, test acc: {:.4f}".format(
                rank, train_acc, val_acc, test_acc
            )
        )
        dist.shutdown()
    # 测试结果一致性
    import numpy as np

    for i in range(len(all_embs) - 1):
        np.testing.assert_array_equal(all_embs[i].numpy(), all_embs[i + 1].numpy())


if __name__ == "__main__":
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
    argparser.add_argument("--batch-size", type=int, default=10000)
    argparser.add_argument("--runs", type=int, default=1)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--dataset", type=str, default="ogbn-products")
    argparser.add_argument(
        "--data-path",
        type=str,
        default="/mnt/ningxin/data/dist",
    )
    argparser.add_argument("--pipeline", action="store_true")
    args = argparser.parse_args()
    print(args)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
