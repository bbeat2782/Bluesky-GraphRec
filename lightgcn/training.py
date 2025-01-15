import torch
import numpy as np
from typing import Tuple
from scipy.sparse import csr_matrix

def sample_triplets(interaction_matrix: csr_matrix, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (user, positive item, negative item) triplets for training."""
    n_users, n_items = interaction_matrix.shape
    
    # Sample users
    users = np.random.randint(0, n_users, batch_size)
    pos_items = []
    neg_items = []
    
    for user in users:
        # Get positive items for user (using sparse matrix)
        pos_items_user = interaction_matrix[user].indices
        # Sample one positive item
        pos_item = np.random.choice(pos_items_user)
        
        # Sample negative item
        while True:
            neg_item = np.random.randint(0, n_items)
            if interaction_matrix[user, neg_item] == 0:
                break
        
        pos_items.append(pos_item)
        neg_items.append(neg_item)
    
    return (torch.LongTensor(users), 
            torch.LongTensor(pos_items), 
            torch.LongTensor(neg_items))

def train_lightgcn(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   train_data: csr_matrix,  # Changed to accept sparse matrix
                   adj_matrix: torch.sparse.FloatTensor,
                   epochs: int = 100,
                   batch_size: int = 1024,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the LightGCN model."""
    model = model.to(device)
    adj_matrix = adj_matrix.to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        user_embeddings, item_embeddings = model(adj_matrix)
        
        # Sample triplets (now works with sparse matrix)
        users, pos_items, neg_items = sample_triplets(train_data, batch_size)
        users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
        
        # Calculate embeddings
        user_emb = user_embeddings[users]
        pos_emb = item_embeddings[pos_items]
        neg_emb = item_embeddings[neg_items]
        
        # Calculate BPR loss
        pos_scores = torch.sum(user_emb * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb * neg_emb, dim=1)
        
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")