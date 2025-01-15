import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, num_layers: int = 3):
        """
        Initialize LightGCN model.
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Dimension of embeddings
            num_layers: Number of GCN layers
        """
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix: torch.sparse.FloatTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            adj_matrix: Sparse adjacency matrix
            
        Returns:
            Tuple of user and item embeddings
        """
        device = adj_matrix.device  # Get device from adj_matrix
        
        # Get initial embeddings and move to correct device
        users_emb = self.user_embedding.weight.to(device)
        items_emb = self.item_embedding.weight.to(device)
        all_emb = torch.cat([users_emb, items_emb])
        
        # Storage for embeddings from all layers
        embs = [all_emb]
        
        # Graph convolution
        for layer in range(self.num_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        
        # Layer combination
        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)
        
        users_emb_final, items_emb_final = torch.split(embs, [self.num_users, self.num_items])
        
        return users_emb_final, items_emb_final