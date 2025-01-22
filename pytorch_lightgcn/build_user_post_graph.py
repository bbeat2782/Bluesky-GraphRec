import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce
import pandas as pd

def build_user_post_graph(likes_df: pd.DataFrame):
    """
    Given a DataFrame of likes with columns [interaction_uri, post_uri, user_uri, timestamp],
    build a bipartite graph (users + posts) for LightGCN.
    """
    # 2.1 Collect unique users and posts
    unique_users = likes_df['user_uri'].unique().tolist()
    unique_posts = likes_df['post_uri'].unique().tolist()
    
    # 2.2 Map user uris to integer IDs, and post uris to integer IDs
    user2id = {u: i for i, u in enumerate(unique_users)}
    post2id = {p: i+len(user2id) for i, p in enumerate(unique_posts)}
    
    num_users = len(user2id)
    num_posts = len(post2id)
    print(f"Number of unique users: {num_users}, number of unique posts: {num_posts}")
    
    # 2.3 Create edges
    # Each 'like' is an edge user -> post. For LightGCN (undirected), we add user->post and post->user
    user_ids = likes_df['user_uri'].apply(lambda x: user2id[x]).values
    post_ids = likes_df['post_uri'].apply(lambda x: post2id[x]).values

    # Combine them into edges: two rows = [source_nodes, target_nodes]
    edge_index = np.vstack((user_ids, post_ids))
    
    # 2.4 Convert to PyTorch tensors
    edge_index = torch.from_numpy(edge_index).long()
    
    # 2.5 Make it undirected for LightGCN
    #   user->post is row 0, post->user is row 1
    #   Then coalesce them (remove duplicates, sort, etc.)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_users + num_posts, num_users + num_posts)
    
    # 2.6 Create a PyG Data object
    data = Data(
        edge_index=edge_index,
        num_nodes=(num_users + num_posts)
    )
    
    # We'll store user/post ID maps in data for possible future use:
    data.user2id = user2id
    data.post2id = post2id
    
    return data