### Ignore this file for now

import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import Embedding
import torch.nn.functional as F

class UltraGCN(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,
        w4: float = 1.0,
        negative_weight: float = 0.5,
        gamma: float = 0.1,
        lambda_: float = 0.1,
        initial_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # UltraGCN specific parameters
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.negative_weight = negative_weight
        self.gamma = gamma
        self.lambda_ = lambda_
        
        # Initialize embeddings
        self.embedding = Embedding(num_nodes, embedding_dim)
        torch.nn.init.normal_(self.embedding.weight, std=initial_weight)

    def forward(
        self,
        edge_index: Adj,
        constraint_mat: OptTensor = None,
        ii_constraint_mat: OptTensor = None,
        ii_neighbor_mat: OptTensor = None,
    ) -> Tensor:
        users = edge_index[0]
        items = edge_index[1]
        
        user_emb = self.embedding(users)
        item_emb = self.embedding(items)
        
        # Basic prediction score
        pred = (user_emb * item_emb).sum(dim=-1)
        
        return pred

    def loss(
        self,
        pred: Tensor,
        pos_edge_index: Adj,
        neg_edge_index: Adj,
        constraint_mat: OptTensor = None,
        ii_constraint_mat: OptTensor = None,
        ii_neighbor_mat: OptTensor = None,
    ) -> Tensor:
        device = pred.device
        users = pos_edge_index[0]
        pos_items = pos_edge_index[1]
        neg_items = neg_edge_index[1]
        
        # Calculate omega weights
        if constraint_mat is not None:
            beta_uD, beta_iD = constraint_mat['beta_uD'].to(device), constraint_mat['beta_iD'].to(device)
            pos_weight = self.w1 + self.w2 * torch.mul(beta_uD[users], beta_iD[pos_items])
            neg_weight = self.w3 + self.w4 * torch.mul(
                torch.repeat_interleave(beta_uD[users], neg_items.size(1)), 
                beta_iD[neg_items.flatten()]
            )
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).to(device)
            neg_weight = self.w3 * torch.ones(len(neg_items)).to(device)
        
        # Main loss
        pos_scores = self(pos_edge_index)
        neg_scores = self(neg_edge_index)
        
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, 
            torch.ones_like(pos_scores),
            weight=pos_weight
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, 
            torch.zeros_like(neg_scores),
            weight=neg_weight
        )
        
        # Item-item loss
        ii_loss = 0
        if ii_neighbor_mat is not None and ii_constraint_mat is not None:
            neighbor_emb = self.embedding(ii_neighbor_mat[pos_items].to(device))
            sim_scores = ii_constraint_mat[pos_items].to(device)
            user_emb = self.embedding(users).unsqueeze(1)
            ii_loss = -sim_scores * (user_emb * neighbor_emb).sum(dim=-1).sigmoid().log().sum()
        
        # L2 regularization
        reg_loss = self.gamma * self.embedding.weight.norm(2).pow(2)
        
        return pos_loss + self.negative_weight * neg_loss + self.lambda_ * ii_loss + reg_loss