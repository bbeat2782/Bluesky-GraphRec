import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.nn.models import LightGCN
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_lightgcn(data, embedding_dim=64, num_layers=1, epochs=1, batch_size=32768):
    """
    Train a LightGCN model on the bipartite graph using BPR loss.
    """
    model = LightGCN(
        num_nodes=data.num_nodes,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create data loader for batch processing
    train_loader = torch.utils.data.DataLoader(
        range(data.edge_index.size(1)),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Track losses for plotting
    epoch_losses = []
    
    # Training loop with progress bar for epochs
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_examples = 0
        
        # Progress bar for batches within each epoch
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}')
        batch_losses = []
        
        for batch_idx in pbar:
            optimizer.zero_grad()
            
            # Get positive edges for this batch
            pos_edge_index = data.edge_index[:, batch_idx].to(device)
            
            # Generate negative samples from item nodes only
            neg_edge_index = torch.stack([
                pos_edge_index[0],
                torch.randint(data.num_users, data.num_nodes,
                            (len(batch_idx),), device=device)
            ], dim=0)
            
            # Combine positive and negative edges
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            
            # Get rankings and split into positive and negative
            pos_rank, neg_rank = model(data.edge_index.to(device), edge_label_index).chunk(2)
            
            # Calculate BPR loss with regularization
            loss = model.recommendation_loss(
                pos_rank,
                neg_rank,
                node_id=edge_label_index.unique(),
            )
            
            loss.backward()
            optimizer.step()
            
            batch_loss = float(loss)
            batch_losses.append(batch_loss)
            total_loss += batch_loss * pos_rank.numel()
            total_examples += pos_rank.numel()
            
            # Update progress bar with current batch loss
            pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})
        
        avg_loss = total_loss / total_examples
        epoch_losses.append(avg_loss)
        
        # Print epoch statistics
        print(f"\nEpoch {epoch:02d} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Min Batch Loss: {min(batch_losses):.4f}")
        print(f"  Max Batch Loss: {max(batch_losses):.4f}")
        print(f"  Loss Std Dev: {np.std(batch_losses):.4f}\n")
    
    return model


def train_lightgcn_v2(data, embedding_dim=64, num_layers=1, epochs=1, batch_size=32768):
    """
    Train a LightGCN model on the bipartite graph using BPR loss.
    """
    model = LightGCN(
        num_nodes=data.num_nodes,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Create data loader for batch processing
    train_loader = torch.utils.data.DataLoader(
        range(data.edge_index.size(1)),
        batch_size=batch_size,
        shuffle=True
    )
    
    # Track losses for plotting
    epoch_losses = []
    
    # Training loop with progress bar for epochs
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_examples = 0
        
        # Progress bar for batches within each epoch
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:02d}')
        batch_losses = []
        
        for batch_idx in pbar:
            optimizer.zero_grad()
            
            # Get positive edges for this batch
            pos_edge_index = data.edge_index[:, batch_idx].to(device)
            
            # Generate negative samples from item nodes only
            neg_edge_index = torch.stack([
                pos_edge_index[0],
                torch.randint(data.num_users, data.num_nodes,
                            (len(batch_idx),), device=device)
            ], dim=0)
            
            # Combine positive and negative edges
            edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            
            # Get rankings and split into positive and negative
            pos_rank, neg_rank = model(data.edge_index.to(device), edge_label_index).chunk(2)
            
            # Calculate BPR loss with regularization
            loss = model.recommendation_loss(
                pos_rank,
                neg_rank,
                node_id=edge_label_index.unique(),
            )
            
            loss.backward()
            optimizer.step()
            
            batch_loss = float(loss)
            batch_losses.append(batch_loss)
            total_loss += batch_loss * pos_rank.numel()
            total_examples += pos_rank.numel()
            
            # Update progress bar with current batch loss
            pbar.set_postfix({'batch_loss': f'{batch_loss:.4f}'})
        
        avg_loss = total_loss / total_examples
        epoch_losses.append(avg_loss)
        
        # Print epoch statistics
        print(f"\nEpoch {epoch:02d} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Min Batch Loss: {min(batch_losses):.4f}")
        print(f"  Max Batch Loss: {max(batch_losses):.4f}")
        print(f"  Loss Std Dev: {np.std(batch_losses):.4f}\n")
    
    return model