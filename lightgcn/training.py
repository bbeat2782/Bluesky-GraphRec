import torch
import numpy as np
from typing import Tuple
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange

def sample_triplets(interaction_matrix: csr_matrix, batch_size: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample (user, positive item, negative item) triplets for training."""
    n_users, n_items = interaction_matrix.shape
    
    # Find users who have at least one interaction
    users_with_interactions = np.where(np.asarray(interaction_matrix.sum(axis=1)).flatten() > 0)[0]
    if len(users_with_interactions) == 0:
        raise ValueError("No users with interactions found in the dataset")
    
    # Sample users (only from those with interactions)
    users = np.random.choice(users_with_interactions, batch_size)
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
                   train_data: csr_matrix,
                   adj_matrix: torch.sparse.FloatTensor,
                   epochs: int = 100,
                   batch_size: int = 1024,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the LightGCN model."""
    model = model.to(device)
    adj_matrix = adj_matrix.to(device)
    
    # Calculate number of batches per epoch
    n_users = train_data.shape[0]
    n_batches = n_users // batch_size + (1 if n_users % batch_size != 0 else 0)
    
    # Create progress bar for epochs
    epoch_bar = trange(epochs, desc="Training")
    for epoch in epoch_bar:
        model.train()
        total_loss = 0
        
        # Create progress bar for batches
        batch_bar = tqdm(range(n_batches), desc=f"Epoch {epoch}", leave=False)
        for batch in batch_bar:
            optimizer.zero_grad()
            
            # Sample triplets (now works with sparse matrix)
            users, pos_items, neg_items = sample_triplets(train_data, batch_size)
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            
            # Forward pass for this batch
            user_embeddings, item_embeddings = model(adj_matrix)
            
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
            
            current_loss = loss.item()
            total_loss += current_loss
            
            # Update batch progress bar
            batch_bar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        # Calculate and display average loss for the epoch
        avg_loss = total_loss / n_batches
        epoch_bar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

def train_ultragcn(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   train_data: csr_matrix,
                   epochs: int = 100,
                   batch_size: int = 1024,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                   lambda_1: float = 1e-3,  # L2 regularization
                   gamma: float = 1e-4):    # Constraint weight
    """Train the UltraGCN model."""
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Sample triplets
        users, pos_items, neg_items = sample_triplets(train_data, batch_size)
        users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
        
        optimizer.zero_grad()
        
        # Calculate positive and negative scores
        pos_scores = model(users, pos_items)
        neg_scores = model(users, neg_items)
        
        # BPR loss
        bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # L2 regularization
        l2_loss = lambda_1 * (
            torch.norm(model.user_embedding(users)) +
            torch.norm(model.item_embedding(pos_items)) +
            torch.norm(model.item_embedding(neg_items))
        )
        
        # Constraint loss (simplified version)
        user_emb = model.user_embedding(users)
        pos_emb = model.item_embedding(pos_items)
        constraint_loss = gamma * torch.mean(torch.abs(torch.norm(user_emb, dim=1) - torch.norm(pos_emb, dim=1)))
        
        # Total loss
        loss = bpr_loss + l2_loss + constraint_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")