import torch
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch_geometric.nn.models import LightGCN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_lightgcn(data, embedding_dim=64, num_layers=2, epochs=5):
    """
    Train a LightGCN model on the bipartite graph using BPR loss.
    """
    model = LightGCN(
        num_nodes=data.num_nodes,
        embedding_dim=embedding_dim,
        num_layers=num_layers
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Generate negative samples on CPU first
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index.cpu(),
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1)
        ).to(device)
        
        # Get rankings for positive and negative edges
        pos_edge_rank = model(data.edge_index.to(device))
        neg_edge_rank = model(neg_edge_index)
        
        # Calculate BPR loss
        loss = model.recommendation_loss(pos_edge_rank, neg_edge_rank)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch:02d}, Loss: {loss.item():.4f}")
    
    return model