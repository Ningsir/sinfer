from typing import Optional, Tuple, NamedTuple, List
import torch as th
from torch_sparse import SparseTensor


def sample_adj(
    rowptr, col, subset: th.Tensor, num_neighbors: int, replace: bool = False
):
    rowptr, col, n_id, e_id = th.ops.torch_sparse.sample_adj(
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
    e_id: Optional[th.Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class SinferPygDataloader(th.utils.data.DataLoader):
    """
    Sampler for the baseline. We modified NeighborSampler class of PyG.

    Parameters
    ----------
        indptr (Tensor): the indptr tensor.
        indices (Tensor): the (memory-mapped) indices tensor.
        fan_outs ([int]): The number of neighbors to sample for each node in each layer.
            If set to `fan_outs[l] = -1`, all neighbors are included in layer `l`.
        node_idx (Tensor): The nodes that should be considered for creating mini-batches.
        **kwargs (optional): Additional arguments of
            `torch.utils.data.DataLoader`, such as `batch_size`,
            `shuffle`, `drop_last`, `num_workers`.
    """

    def __init__(
        self, indptr, indices, fan_outs: List[int], node_idx: th.Tensor, **kwargs
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]
        if "dataset" in kwargs:
            del kwargs["dataset"]
        if "shuffle" in kwargs:
            del kwargs["shuffle"]

        self.indptr = indptr
        self.indices = indices
        self.node_idx = node_idx
        self.fan_outs = fan_outs

        if node_idx.dtype == th.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super(SinferPygDataloader, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, shuffle=False, **kwargs
        )

    def sample(self, batch):
        """
        采样

        Parameters
        ----------
            batch: 被采样的种子节点

        Ouptut
        ----------
        (batch_size, batch, n_id, adjs):
            batch: 种子点
            n_id: 采样子图包含的所有节点
            adjs: 采样得到的子图
        """
        if not isinstance(batch, th.Tensor):
            batch = th.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.fan_outs:
            adj_t, n_id = sample_adj(self.indptr, self.indices, n_id, size, False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]

            adjs.append(Adj(adj_t, e_id, size))

        adjs = adjs[::-1]
        out = (batch_size, batch, n_id, adjs)
        return out

    def __repr__(self):
        return "{}(sizes={})".format(self.__class__.__name__, self.fan_outs)
