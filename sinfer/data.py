import os
import json
import torch as th
import numpy as np


class SinferDataset:
    def __init__(self, path):
        self.coo_path = os.path.join(path, "coo.bin")
        self.features_path = os.path.join(path, "feat.bin")
        self.offsets_path = os.path.join(path, "offsets.txt")
        self.conf_path = os.path.join(path, "conf.json")
        self.indptr_path = os.path.join(path, "indptr.bin")
        self.indices_path = os.path.join(path, "indices.bin")
        self.labels_path = os.path.join(path, "labels.bin")
        self.train_idx_path = os.path.join(path, "train_idx.bin")
        self.val_idx_path = os.path.join(path, "val_idx.bin")
        self.test_idx_path = os.path.join(path, "test_idx.bin")

        self.conf = json.load(open(self.conf_path, "r"))
        self._num_nodes = self.conf["num_nodes"]
        self._feat_dim = self.conf["feat_dim"]
        self._num_classes = self.conf["num_classes"]

    def coo(self):
        coo = np.fromfile(self.coo_path, dtype=np.int64)
        coo = coo.reshape(2, -1)
        coo_row = th.from_numpy(coo[0])
        coo_col = th.from_numpy(coo[1])
        return coo_row, coo_col

    @property
    def feat_path(self):
        return self.features_path

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def feat_dim(self):
        return self._feat_dim

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def offsets(self):
        offsets = np.fromfile(self.offsets_path, sep="\n", dtype=np.int64)
        return list(offsets)

    @property
    def indptr(self):
        indptr_ = np.fromfile(self.indptr_path, dtype=self.conf["indptr_dtype"])
        indptr_ = th.from_numpy(indptr_)
        return indptr_

    @property
    def indices(self):
        indices_ = np.fromfile(self.indices_path, dtype=self.conf["indices_dtype"])
        indices_ = th.from_numpy(indices_)
        return indices_

    @property
    def labels(self):
        labels_ = np.fromfile(self.labels_path, dtype=self.conf["labels_dtype"])
        labels_ = th.from_numpy(labels_)
        return labels_

    @property
    def train_idx(self):
        train_idx_ = np.fromfile(
            self.train_idx_path, dtype=self.conf["train_idx_dtype"]
        )
        train_idx_ = th.from_numpy(train_idx_)
        return train_idx_

    @property
    def val_idx(self):
        val_idx_ = np.fromfile(self.val_idx_path, dtype=self.conf["val_idx_dtype"])
        val_idx_ = th.from_numpy(val_idx_)
        return val_idx_

    @property
    def test_idx(self):
        test_idx_ = np.fromfile(self.test_idx_path, dtype=self.conf["test_idx_dtype"])
        test_idx_ = th.from_numpy(test_idx_)
        return test_idx_

    def __str__(self) -> str:
        return "num_nodes: {}, feat_dim: {}, num_classes: {}".format(
            self.num_nodes, self.feat_dim, self.num_classes
        )
