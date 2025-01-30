import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce
from temporal_split import create_split_with_time
import pandas as pd
import pickle

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
    # turns the actual uris into numerical ids using user2id and post2id
    user_ids = likes_df['user_uri'].apply(lambda x: user2id[x]).values
    post_ids = likes_df['post_uri'].apply(lambda x: post2id[x]).values

    # Combine them into edges: two rows = [source_nodes, target_nodes]
    # Visualization (top row = user_ids, bottom row = post_ids):
    # tensor([[     0,      1,      1,  ..., 846523, 846524, 846525],
    #    [ 30470,  30471,  30472,  ...,  13554,  19451,  25516]])
    edge_index = np.vstack((user_ids, post_ids))
    
    # 2.4 Convert to PyTorch tensors
    edge_index = torch.from_numpy(edge_index).long()
    
    # 2.5 Make it undirected for LightGCN
    #   user->post is row 0, post->user is row 1, has the effect of doubling the number of edges
    #   Then coalesce them (remove duplicates, sort, etc.)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_users + num_posts, num_users + num_posts)
    
    # 2.6 Create a PyG Data object
    data = Data(
        edge_index=edge_index,
        num_nodes=(num_users + num_posts)
    )
    
    # Add these attributes needed for training
    data.num_users = num_users
    data.num_items = num_posts  # Also adding this for consistency
    
    # We'll store user/post ID maps in data for possible future use:
    data.user2id = user2id
    data.post2id = post2id
    
    return data

def convert_timestamp_to_float(timestamp):
    """Convert pandas.Timestamp to float in the format YYYYMMDDHHMMSS"""
    return float(timestamp.strftime("%Y%m%d%H%M%S"))

def build_user_post_graph_v2(likes_df: pd.DataFrame):
    """
    Given a DataFrame of likes with columns [user_id, post_id, timestamp],
    build a bipartite graph (users + posts) for LightGCN.
    """
    user_mapping_path = '/home/sgan/private/DyGLib/DG_data/bluesky/user_mapping.pkl'
    post_mapping_path = '/home/sgan/private/DyGLib/DG_data/bluesky/post_mapping.pkl'

    with open(user_mapping_path, 'rb') as f:
        user_mapping = pickle.load(f)
    with open(post_mapping_path, 'rb') as f:
        post_mapping = pickle.load(f)

    # Map `formatted_postUri` and `user_uri` using the mappings
    likes_df['user_id'] = likes_df['user_uri'].map(user_mapping)
    likes_df['post_id'] = likes_df['formatted_postUri'].map(post_mapping)
    user_max_id = likes_df['user_id'].max() + 1
    likes_df['post_id'] = likes_df['post_id'] + user_max_id

    likes_df = likes_df[['user_id', 'post_id', 'timestamp']]
    likes_df.loc[:,'timestamp'] = likes_df['timestamp'].apply(convert_timestamp_to_float)
    likes_df = likes_df.sort_values(by='timestamp')
    likes_df = likes_df.reset_index(drop=True)
    
    # 2.1 Collect unique users and posts
    unique_users = likes_df['user_id'].unique().tolist()
    unique_posts = likes_df['post_id'].unique().tolist()
        
    num_users = len(unique_users)
    num_posts = len(unique_posts)
    print(f"Number of unique users: {num_users}, number of unique posts: {num_posts}")
    
    # 2.3 Create edges
    # turns the actual uris into numerical ids using user2id and post2id
    # user_ids = likes_df['user_id'].values
    # post_ids = likes_df['post_id'].values


    # Assign labels: 0 = train, 1 = val, 2 = test
    train_df, val_df, test_df = create_split_with_time(likes_df)
    train_edges = torch.zeros(len(train_df), dtype=torch.long)
    val_edges = torch.ones(len(val_df), dtype=torch.long)
    test_edges = torch.full((len(test_df),), 2, dtype=torch.long)
    
    # Concatenate edges
    labels = torch.cat([train_edges, val_edges, test_edges], dim=0)
    # Convert user and post IDs from Series to Tensors
    train_user_ids = torch.tensor(train_df['user_id'].values, dtype=torch.long)
    val_user_ids = torch.tensor(val_df['user_id'].values, dtype=torch.long)
    test_user_ids = torch.tensor(test_df['user_id'].values, dtype=torch.long)
    
    train_post_ids = torch.tensor(train_df['post_id'].values, dtype=torch.long)
    val_post_ids = torch.tensor(val_df['post_id'].values, dtype=torch.long)
    test_post_ids = torch.tensor(test_df['post_id'].values, dtype=torch.long)

    num_train_users = torch.unique(train_user_ids).size(0)
    num_train_posts = torch.unique(train_post_ids).size(0)
    num_val_users = torch.unique(val_user_ids).size(0)
    num_val_posts = torch.unique(val_post_ids).size(0)
    num_test_users = torch.unique(test_user_ids).size(0)
    num_test_posts = torch.unique(test_post_ids).size(0)
    
    # Concatenate tensors
    user_ids = torch.cat([train_user_ids, val_user_ids, test_user_ids])
    post_ids = torch.cat([train_post_ids, val_post_ids, test_post_ids])
    
    # Create edge_index with labels
    edge_index = torch.vstack((user_ids, post_ids))
    
    # Convert edges to tuples and map them to their labels
    edge_to_label = {}
    for i in range(edge_index.size(1)):
        edge = tuple(edge_index[:, i].tolist())
        edge_to_label[edge] = labels[i].item()
        edge_to_label[(edge[1], edge[0])] = labels[i].item()  # Add reverse edge
    
    # 2.5 Make it undirected for LightGCN
    #   user->post is row 0, post->user is row 1, has the effect of doubling the number of edges
    #   Then coalesce them (remove duplicates, sort, etc.)
    edge_index = to_undirected(edge_index)
    edge_index, _ = coalesce(edge_index, None, num_users + num_posts, num_users + num_posts)

    undirected_labels = []
    for i in range(edge_index.size(1)):
        edge = tuple(edge_index[:, i].tolist())
        label = edge_to_label.get(edge)
        if label is None:
            raise ValueError(f"Label for edge {edge} not found!")
        undirected_labels.append(label)

    undirected_labels = torch.tensor(undirected_labels, dtype=torch.long)
    print(len(undirected_labels))
    train_mask = undirected_labels == 0  # Training edges
    val_mask = undirected_labels == 1    # Validation edges
    test_mask = undirected_labels == 2   # Test edges

    # Training edges
    train_edge_index = edge_index[:, train_mask]
    
    # Validation edges
    val_edge_index = edge_index[:, val_mask]
    
    # Test edges
    test_edge_index = edge_index[:, test_mask]
    
    # 2.6 Create a PyG Data object
    data = Data(
        edge_index=edge_index,
        num_nodes=(num_users + num_posts)
    )
    
    # Add these attributes needed for training
    data.num_users = num_users
    data.num_items = num_posts  # Also adding this for consistency
    
    # We'll store user/post ID maps in data for possible future use:
    #data.user2id = user2id
    #data.post2id = post2id
    data.edge_labels = undirected_labels
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    data.num_train_users = num_train_users
    data.num_train_posts = num_train_posts
    data.num_val_users = num_val_users
    data.num_val_posts = num_val_posts
    data.num_test_users = num_test_users
    data.num_test_posts = num_test_posts
    
    return data