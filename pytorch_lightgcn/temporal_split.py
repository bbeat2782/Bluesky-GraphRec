import pandas as pd
import torch
from torch_geometric.data import Data
from build_user_post_graph import build_user_post_graph

def create_temporal_split(likes_df: pd.DataFrame, split_ratio: float = 0.8) -> tuple[Data, Data, dict]:
    """
    Create temporal train/test split of the interaction data.
    
    Args:
        likes_df: DataFrame with columns [interaction_uri, post_uri, user_uri, timestamp]
        split_ratio: Ratio of data to use for training (0.8 = 80% train, 20% test)
    
    Returns:
        train_data: PyG Data object for training
        test_data: PyG Data object for testing
        test_interactions: Dictionary mapping user_ids to lists of their test post_ids
    """
    # Sort by timestamp to ensure temporal split
    likes_df = likes_df.sort_values('timestamp')
    
    # Find split timestamp
    split_idx = int(len(likes_df) * split_ratio)
    split_timestamp = likes_df.iloc[split_idx]['timestamp']
    
    # Split into train and test
    train_df = likes_df[likes_df['timestamp'] < split_timestamp]
    test_df = likes_df[likes_df['timestamp'] >= split_timestamp]
    
    # Build train graph
    train_data = build_user_post_graph(train_df)
    
    # Filter test set to only include users and posts that existed in training
    train_users = set(train_data.user2id.keys())
    train_posts = set(train_data.post2id.keys())
    
    test_df = test_df[
        test_df['user_uri'].isin(train_users) & 
        test_df['post_uri'].isin(train_posts)
    ]
    
    # Build test graph using same node mappings as train
    test_data = build_user_post_graph(test_df)
    test_data.user2id = train_data.user2id  # Use same mappings as train
    test_data.post2id = train_data.post2id
    
    # Create dictionary of test interactions for evaluation
    test_interactions = {}
    for _, row in test_df.iterrows():
        user_id = train_data.user2id[row['user_uri']]
        post_id = train_data.post2id[row['post_uri']]
        if user_id not in test_interactions:
            test_interactions[user_id] = []
        test_interactions[user_id].append(post_id)
    
    print(f"Train interactions: {len(train_df)}")
    print(f"Test interactions: {len(test_df)}")
    print(f"Split timestamp: {split_timestamp}")
    print(f"Users with test interactions: {len(test_interactions)}")
    
    return train_data, test_data, test_interactions