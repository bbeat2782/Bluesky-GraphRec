from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


class LGConv(MessagePassing):
    r"""The Light Graph Convolution (LGC) operator from the `"LightGCN:
    Simplifying and Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126>`_ paper.

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        \frac{e_{j,i}}{\sqrt{\deg(i)\deg(j)}} \mathbf{x}_j

    Args:
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be normalized via symmetric normalization.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    def __init__(self, normalize: bool = True, **kwargs):
        # Set default aggregation to 'add' if not specified
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.normalize = normalize

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        # x shape: [num_nodes, num_features]
        # edge_index shape: [2, num_edges] 
        # edge_weight shape (if provided): [num_edges]

        if self.normalize and isinstance(edge_index, Tensor):
            # Compute symmetric normalization using node degrees:
            # edge_weight_norm = edge_weight / sqrt(deg(src) * deg(dst))
            out = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                           add_self_loops=False, flow=self.flow, dtype=x.dtype)
            edge_index, edge_weight = out
        elif self.normalize and isinstance(edge_index, SparseTensor):
            # Same normalization but for sparse tensor format
            edge_index = gcn_norm(edge_index, None, x.size(self.node_dim),
                                  add_self_loops=False, flow=self.flow,
                                  dtype=x.dtype)

        # Propagate messages along edges
        # Output shape: [num_nodes, num_features]
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # x_j shape: [num_edges, num_features] - features of source nodes
        # edge_weight shape (if provided): [num_edges]
        # Returns weighted source node features
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        # Fused message passing and aggregation for better efficiency
        # adj_t: Sparse adjacency matrix
        # x shape: [num_nodes, num_features]
        # Returns aggregated messages shape: [num_nodes, num_features]
        return spmm(adj_t, x, reduce=self.aggr)