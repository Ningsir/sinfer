import os
import json
import torch as th
import numpy as np


class SinferDataset():
    def __init__(self, path):
        self.coo_path = os.path.join(path, 'coo.bin')
        self.features_path = os.path.join(path, 'feat.bin')
        self.offsets_path = os.path.join(path, 'offsets.txt')
        self.conf_path = os.path.join(path, 'conf.json')
         
        self.conf = json.load(open(self.conf_path, 'r'))
        self._num_nodes = self.conf['num_nodes']
        self._feat_dim = self.conf['feat_dim']
        self._num_classes = self.conf['num_classes']

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
        offsets = np.fromfile(self.offsets_path, sep='\n', dtype=np.int64)
        return list(offsets)


    def __str__(self) -> str:
        return "num_nodes: {}, feat_dim: {}, num_classes: {}".format(self.num_nodes, self.feat_dim, self.num_classes)
