import torch


global_node_id = torch.Tensor
local_node_id = torch.Tensor


class PartInfo:
    """
    分区信息: 利用节点度的信息来判断该节点是边界点还是内节点,
    `local_degree[local_id] == global_degree[local_id]`表示节点`local_id`是内节点, 否则是边界点.
    边界点分为master节点和mirror节点:
        1. master节点: global_node_id % rank == 0;
        2. mirror节点: global_node_id % rank != 0

    Parameters
    ----------
    local_nodes: 存储在本地的节点, 从0到n-1的连续ID
    global_nodes: 全局的原始节点ID
    local_degree: 本地子图中节点的度, 利用local_id进行访问
    global_degree: 全局图中节点的度, 利用local_id进行访问
    """

    def __init__(
        self,
        local_nodes: torch.Tensor,
        global_nodes: torch.Tensor,
        local_degree: torch.Tensor,
        global_degree: torch.Tensor,
    ):
        self.local_nodes = local_nodes
        self.global_nodes = global_nodes
        self.local_degree = local_degree
        self.global_degree = global_degree
        # local id
        self.local_boundary_nodes = torch.nonzero(
            local_degree != global_degree, as_tuple=True
        )[0]

    def get_boundary_nodes(
        self, local_nodes: torch.Tensor, output_index: bool = True
    ) -> local_node_id:
        """
        获取local_nodes中的所有边界点, 如果output_index=True, 则还会输出边界点在local_nodes中的对应索引
        """
        mask = self.local_degree[local_nodes] != self.global_degree[local_nodes]
        if not output_index:
            return local_nodes.masked_select(mask)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        return indices, local_nodes.masked_select(mask)

    def get_local_degree(self, local_nodes):
        return self.local_degree[local_nodes]

    def get_global_degree(self, local_nodes):
        return self.global_degree[local_nodes]

    def get_all_boundary_nodes(self) -> local_node_id:
        return self.local_boundary_nodes

    def get_all_inner_nodes(self) -> local_node_id:
        return torch.nonzero(self.local_degree == self.global_degree, as_tuple=True)[0]

    def get_inner_nodes(
        self, local_nodes: torch.Tensor, output_index: bool = True
    ) -> local_node_id:
        mask = self.local_degree[local_nodes] == self.global_degree[local_nodes]
        if not output_index:
            return local_nodes.masked_select(mask)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        return indices, local_nodes.masked_select(mask)

    def local_to_global_id(self, local_nodes) -> global_node_id:
        return self.global_nodes[local_nodes]

    def global_to_local_id(self, global_nodes) -> local_node_id:
        raise NotImplementedError("global_to_local_id not implemented!")
