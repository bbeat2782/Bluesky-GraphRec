import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def factorize(coo_matrix, n_components=128, n_clusters=100, device='cuda'):
    """
    Factorize the input matrix using PyTorch's SVD implementation.
    
    Args:
        coo_matrix: scipy sparse COO matrix of shape (n_consumers, n_producers)
        n_components: dimensionality of the embedding space
        n_clusters: number of producer communities
        device: 'cuda' or 'cpu'
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
    # Note: V is already the right shape (n_consumers, n_components)
    consumer_embeddings = (V * sqrt_S.unsqueeze(0)).cpu().numpy()
    
    # L2 normalize both embeddings before clustering
    producer_embeddings_norm = normalize(producer_embeddings, norm='l2')
    consumer_embeddings_norm = normalize(consumer_embeddings, norm='l2')
    
    # Cluster the normalized producer embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    producer_communities = kmeans.fit_predict(producer_embeddings_norm)
    
    # Calculate affinity scores (0-1) for how strongly each producer belongs to their assigned cluster
    assigned_distances = np.zeros(len(producer_embeddings_norm))
    for i, (producer, cluster) in enumerate(zip(producer_embeddings_norm, producer_communities)):
        distance = np.linalg.norm(producer - kmeans.cluster_centers_[cluster])
        assigned_distances[i] = distance
    
    producer_community_affinities = 1 - (assigned_distances / assigned_distances.max())
    
    return producer_communities, producer_community_affinities, consumer_embeddings_norm, producer_embeddings_norm, kmeans.cluster_centers_




# ############################################################
# IGNORE EVERYTHING BELOW
# ############################################################

# from torchnmf.nmf import NMF
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import normalize
# import numpy as np


# def factorize_torch(matrix, alpha=0.0, n_components=20, n_clusters=100, device='cuda'):
#     """
#     Factorize using TorchNMF and get intermediate embeddings
    
#     Args:
#         matrix: scipy sparse matrix
#         algorithm: 'nmf' or 'svd'
#         n_components: dimensionality of intermediate space
#         n_clusters: number of final communities
#         device: 'cuda' or 'cpu'
#     """
#     # Convert sparse matrix to torch tensor
#     V = torch.from_numpy(matrix.T.toarray()).float().to(device)
    
#     # Initialize NMF model
#     model = NMF(V.shape, n_components).to(device)
    
#     # Fit the model with L1 regularization
#     model.fit(
#         V,
#         beta=1,  # KL divergence
#         tol=1e-4,
#         max_iter=200,
#         verbose=True,
#         alpha=alpha,  # Similar to alpha_W in sklearn
#         l1_ratio=1.0
#     )
    
#     # Get producer embeddings (equivalent to fit_transform in sklearn)
#     producer_embeddings = model.H.detach().cpu().numpy()
    
#     # Get consumer embeddings through matrix multiplication
#     consumer_embeddings = matrix @ producer_embeddings

#     # L1 normalize both embeddings
#     producer_embeddings_norm = normalize(producer_embeddings, norm='l1')
#     consumer_embeddings_norm = normalize(consumer_embeddings, norm='l1')

#     # Cluster the normalized producer embeddings
#     kmeans = KMeans(n_clusters=n_clusters)
#     producer_communities = kmeans.fit_predict(producer_embeddings_norm)

#     # Calculate affinities
#     assigned_distances = np.zeros(len(producer_embeddings_norm))
#     for i, (producer, cluster) in enumerate(zip(producer_embeddings_norm, producer_communities)):
#         distance = np.linalg.norm(producer - kmeans.cluster_centers_[cluster])
#         assigned_distances[i] = distance

#     producer_community_affinities = 1 - (assigned_distances / assigned_distances.max())

#     return producer_community_affinities, consumer_embeddings_norm, kmeans.cluster_centers_

# # Try it out
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# producer_community_affinities, consumer_embeddings, kmeans_cluster_centers = factorize_torch(
#     matrix,
#     n_components=20,
#     n_clusters=100,
#     device=device
# )

# # Print some stats
# print(f"Producer affinities shape: {producer_community_affinities.shape}")
# print(f"Consumer embeddings shape: {consumer_embeddings.shape}")
# print(f"Cluster centers shape: {kmeans_cluster_centers.shape}")