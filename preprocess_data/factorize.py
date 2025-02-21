import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.linalg import orthogonal_procrustes


def align_embeddings(source_consumer, source_producer, target_consumer):
    """
    Align both consumer and producer embeddings using the same rotation matrix.
    Rotation matrix is computed from consumer embeddings.
    
    Args:
        source_consumer: consumer embeddings to align (assumed same shape as target_consumer)
        source_producer: producer embeddings to align (or None)
        target_consumer: target consumer embeddings to align to
    Returns:
        aligned_consumer, aligned_producer: aligned embeddings
    """
    # Calculate optimal rotation matrix using consumer embeddings
    R, _ = orthogonal_procrustes(source_consumer, target_consumer)
    
    # Apply rotation to consumer embeddings
    aligned_consumer = source_consumer @ R
    
    # Handle case where source_producer is None
    if source_producer is None:
        aligned_producer = None
    else:
        aligned_producer = source_producer @ R
    
    return aligned_consumer, aligned_producer


def factorize(coo_matrix, n_components=128, n_clusters=100, device='cuda', previous_consumer_embeddings=None):
    """
    Factorize the input matrix using PyTorch's SVD implementation.
    
    Args:
        coo_matrix: scipy sparse COO matrix of shape (n_consumers, n_producers)
        n_components: dimensionality of the embedding space
        n_clusters: number of producer communities
        device: 'cuda' or 'cpu'
        previous_consumer_embeddings: consumer embeddings from previous day for alignment
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
    if previous_consumer_embeddings is not None:
        prev_consumer_emb = previous_consumer_embeddings
        consumer_embeddings_norm, producer_embeddings_norm = align_embeddings(
            consumer_embeddings_norm,
            producer_embeddings_norm,
            prev_consumer_emb
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



