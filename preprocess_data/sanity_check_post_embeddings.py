import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import argparse
import random

def sanity_check_post_embeddings():
    # Paths
    processed_data_path = "../processed_data/bluesky"
    interaction_file = os.path.join(processed_data_path, "ml_bluesky.csv")
    post_embeddings_file = os.path.join(os.path.expanduser("~"), 'post_dynamic_embeddings.parquet')
    
    print("Loading data files...")
    # Load interaction data
    interactions = pd.read_csv(interaction_file)
    
    # Load post embeddings
    try:
        post_embeddings = pd.read_parquet(post_embeddings_file)
        print(f"Successfully loaded post embeddings of shape: {post_embeddings.shape}")
    except Exception as e:
        print(f"Error loading post embeddings: {e}")
        return
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Total interactions: {len(interactions)}")
    print(f"Unique posts in interactions: {interactions['i'].nunique()}")
    print(f"Unique users in interactions: {interactions['u'].nunique()}")
    print(f"Total post embeddings: {len(post_embeddings)}")
    print(f"Unique posts in embeddings: {post_embeddings['post_id'].nunique()}")
    
    if 'user_id' in post_embeddings.columns:
        print(f"Unique users in embeddings: {post_embeddings['user_id'].nunique()}")
    
    # Check embedding dimensions
    if 'embedding' in post_embeddings.columns:
        sample_embedding = post_embeddings['embedding'].iloc[0]
        if isinstance(sample_embedding, (list, np.ndarray)):
            print(f"Embedding dimension: {len(sample_embedding)}")
        else:
            print(f"Warning: Embeddings not in expected format: {type(sample_embedding)}")
    
    # Check post ID overlap
    interaction_post_ids = set(interactions['i'].unique())
    embedding_post_ids = set(post_embeddings['post_id'].unique())
    
    common_posts = interaction_post_ids.intersection(embedding_post_ids)
    print(f"\n=== Post Coverage ===")
    print(f"Posts in both datasets: {len(common_posts)} ({len(common_posts)/len(interaction_post_ids)*100:.2f}% of interaction posts)")
    
    missing_posts = interaction_post_ids - embedding_post_ids
    print(f"Posts in interactions but missing from embeddings: {len(missing_posts)}")
    if len(missing_posts) > 0 and len(missing_posts) < 10:
        print(f"Missing posts: {list(missing_posts)}")
    elif len(missing_posts) >= 10:
        print(f"Sample of missing posts: {random.sample(list(missing_posts), 10)}")
    
    extra_posts = embedding_post_ids - interaction_post_ids
    print(f"Posts in embeddings but not in interactions: {len(extra_posts)}")
    if len(extra_posts) > 0 and len(extra_posts) < 10:
        print(f"Extra posts: {list(extra_posts)}")
    elif len(extra_posts) >= 10:
        print(f"Sample of extra posts: {random.sample(list(extra_posts), 10)}")
    
    # Check user ID overlap if user_id is present
    if 'user_id' in post_embeddings.columns:
        interaction_user_ids = set(interactions['u'].unique())
        embedding_user_ids = set(post_embeddings['user_id'].unique())
        
        common_users = interaction_user_ids.intersection(embedding_user_ids)
        print(f"\n=== User Coverage ===")
        print(f"Users in both datasets: {len(common_users)} ({len(common_users)/len(interaction_user_ids)*100:.2f}% of interaction users)")
        
        missing_users = interaction_user_ids - embedding_user_ids
        print(f"Users in interactions but missing from embeddings: {len(missing_users)}")
        if len(missing_users) > 0 and len(missing_users) < 10:
            print(f"Missing users: {list(missing_users)}")
        elif len(missing_users) >= 10:
            print(f"Sample of missing users: {random.sample(list(missing_users), 10)}")
    
    # Check timestamp ranges
    print("\n=== Timestamp Analysis ===")
    # Convert interaction timestamps
    interactions['timestamp'] = pd.to_datetime(interactions['ts'], unit='s')
    
    # Convert embedding timestamps if needed
    if not pd.api.types.is_datetime64_any_dtype(post_embeddings['timestamp']):
        try:
            post_embeddings['timestamp'] = pd.to_datetime(post_embeddings['timestamp'])
        except:
            print("Warning: Could not convert embedding timestamps to datetime")
    
    int_time_min = interactions['timestamp'].min()
    int_time_max = interactions['timestamp'].max()
    emb_time_min = post_embeddings['timestamp'].min()
    emb_time_max = post_embeddings['timestamp'].max()
    
    print(f"Interaction time range: {int_time_min} to {int_time_max}")
    print(f"Embedding time range: {emb_time_min} to {emb_time_max}")
    
    # Check if ranges overlap properly
    if emb_time_min <= int_time_max and emb_time_max >= int_time_min:
        print("✅ Time ranges overlap correctly")
    else:
        print("❌ Warning: Time ranges do not overlap properly")
    
    # Sample random posts to check embedding counts
    print("\n=== Sample Post Analysis ===")
    sample_size = min(10, len(common_posts))
    sample_posts = random.sample(list(common_posts), sample_size)
    
    for post_id in sample_posts:
        post_interactions = interactions[interactions['i'] == post_id]
        post_embeddings_count = len(post_embeddings[post_embeddings['post_id'] == post_id])
        
        interaction_count = len(post_interactions)
        ratio = post_embeddings_count / interaction_count if interaction_count > 0 else 0
        
        status = "✅" if 0.5 <= ratio <= 1.5 else "❓"
        print(f"{status} Post {post_id}: {post_embeddings_count} embeddings / {interaction_count} interactions ({ratio:.2f})")
    
    print("\n=== Summary ===")
    if len(common_posts) / len(interaction_post_ids) > 0.9:
        print("✅ Most interaction posts have embeddings")
    else:
        print("❌ Many interaction posts are missing embeddings")
    
    if len(missing_posts) == 0:
        print("✅ All interaction posts have embeddings")
    else:
        print("❌ Some interaction posts are missing embeddings")
    
    if len(extra_posts) == 0:
        print("✅ No extra post embeddings found")
    else:
        print("❓ Some post embeddings don't match interaction posts")
    
    if 'user_id' in post_embeddings.columns and len(common_users) / len(interaction_user_ids) > 0.9:
        print("✅ Most interaction users are represented in the embeddings")
    
    print("\nDone with post embeddings sanity check!")

if __name__ == "__main__":
    sanity_check_post_embeddings()