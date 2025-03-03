import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.linalg import orthogonal_procrustes
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
torch.use_deterministic_algorithms(True)

def align_embeddings(source_consumer, source_producer, target_consumer, consumer_ids=None, prev_consumer_ids=None):
    """
    Align embeddings using only common users between days.
    
    Args:
        source_consumer: new day's consumer embeddings
        source_producer: new day's producer embeddings
        target_consumer: previous day's consumer embeddings
        consumer_ids: current day's consumer IDs
        prev_consumer_ids: previous day's consumer IDs
    """
    if consumer_ids is not None and prev_consumer_ids is not None:
        # Find common users
        common_users = set(consumer_ids) & set(prev_consumer_ids)
        
        # Get indices for common users
        curr_mask = np.array([id in common_users for id in consumer_ids])
        prev_mask = np.array([id in common_users for id in prev_consumer_ids])
        
        # Align using only common users
        R, _ = orthogonal_procrustes(
            source_consumer[curr_mask], 
            target_consumer[prev_mask]
        )
    else:
        # If no IDs provided, assume same users (old behavior)
        R, _ = orthogonal_procrustes(source_consumer, target_consumer)
    
    # Apply rotation to all users
    aligned_consumer = source_consumer @ R
    aligned_producer = source_producer @ R
    
    return aligned_consumer, aligned_producer


def factorize(coo_matrix, n_components=128, n_clusters=100, device='cuda', 
              previous_embeddings=None, consumer_ids=None, prev_consumer_ids=None):
    """
    Factorize the input matrix using PyTorch's SVD implementation.
    
    Args:
        coo_matrix: scipy sparse COO matrix of shape (n_consumers, n_producers)
        n_components: dimensionality of the embedding space
        n_clusters: number of producer communities
        device: 'cuda' or 'cpu'
        previous_embeddings: previous day's embeddings (consumer, producer)
        consumer_ids: current day's consumer IDs
        prev_consumer_ids: previous day's consumer IDs
    """
    
    # Convert scipy COO to torch sparse
    # Note: matrix.T because torch expects (n_features, n_samples)
    values = torch.FloatTensor(coo_matrix.T.data)
    indices = torch.LongTensor(np.vstack((coo_matrix.T.row, coo_matrix.T.col)))
    
    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, 
        torch.Size(coo_matrix.T.shape)
    ).to(device)
    
    # Compute truncated SVD
    U, S, V = torch.svd_lowrank(sparse_tensor, q=n_components)
    
    # Take only the top n_components
    U = U[:, :n_components]
    S = S[:n_components]
    V = V[:, :n_components]
    
    # Compute producer embeddings (U * sqrt(S))
    sqrt_S = torch.sqrt(S)
    producer_embeddings = (U * sqrt_S.unsqueeze(0)).cpu().numpy()
    
    # Compute consumer embeddings (V * sqrt(S))
    consumer_embeddings = (V * sqrt_S.unsqueeze(0)).cpu().numpy()
    
    # L2 normalize both embeddings before clustering
    producer_embeddings_norm = normalize(producer_embeddings, norm='l2')
    consumer_embeddings_norm = normalize(consumer_embeddings, norm='l2')
    
    # If we have previous embeddings, align the new ones (should be everything besides the first day)
    if previous_embeddings is not None:
        prev_producer_emb, prev_consumer_emb = previous_embeddings
        consumer_embeddings_norm, producer_embeddings_norm = align_embeddings(
            consumer_embeddings_norm,
            producer_embeddings_norm,
            prev_consumer_emb,
            consumer_ids,
            prev_consumer_ids
        )
    
    # Cluster the aligned producer embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    producer_communities = kmeans.fit_predict(producer_embeddings_norm)
    
    # Calculate affinity scores (0-1) for how strongly each producer belongs to their assigned cluster
    assigned_distances = np.zeros(len(producer_embeddings_norm))
    for i, (producer, cluster) in enumerate(zip(producer_embeddings_norm, producer_communities)):
        distance = np.linalg.norm(producer - kmeans.cluster_centers_[cluster])
        assigned_distances[i] = distance
    
    producer_community_affinities = 1 - (assigned_distances / assigned_distances.max())
    
    return producer_communities, producer_community_affinities, consumer_embeddings_norm, producer_embeddings_norm, kmeans.cluster_centers_



