import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import argparse
import random

def sanity_check_mappings():
    # Paths
    processed_data_path = "../processed_data/bluesky"
    interaction_file = os.path.join(processed_data_path, "ml_bluesky.csv")
    user_features_file = os.path.join(processed_data_path, "user_dynamic_features.pkl")
    
    print("Loading data files...")
    # Load interaction data
    interactions = pd.read_csv(interaction_file)
    
    # Load user dynamic features
    with open(user_features_file, "rb") as f:
        user_features = pickle.load(f)
    
    # Extract unique user IDs from interactions
    interaction_user_ids = set(interactions['u'].unique())
    print(f"Found {len(interaction_user_ids)} unique users in interaction data")
    
    # Get all user IDs from dynamic features
    feature_user_ids = set()
    for date, users in user_features.items():
        feature_user_ids.update(users.keys())
    print(f"Found {len(feature_user_ids)} unique users in dynamic features")
    
    # Check overlap
    common_users = interaction_user_ids.intersection(feature_user_ids)
    print(f"Users in both datasets: {len(common_users)} ({len(common_users)/len(interaction_user_ids)*100:.2f}% of interaction users)")
    
    # Check for missing users
    missing_users = interaction_user_ids - feature_user_ids
    print(f"Users in interactions but missing from features: {len(missing_users)}")
    if len(missing_users) > 0:
        print(f"Sample of missing users: {list(missing_users)[:5]}")
    
    extra_users = feature_user_ids - interaction_user_ids
    print(f"Users in features but not in interactions: {len(extra_users)}")
    if len(extra_users) > 0:
        print(f"Sample of extra users: {list(extra_users)[:5]}")
    
    # Check embedding dimensions
    if feature_user_ids:
        sample_date = list(user_features.keys())[0]
        sample_user = list(user_features[sample_date].keys())[0]
        embedding_dim = len(user_features[sample_date][sample_user])
        print(f"Embedding dimension: {embedding_dim}")
    
    # Check time range
    if user_features:
        dates = [datetime.fromtimestamp(ts) for ts in user_features.keys()]
        print(f"Time range of features: {min(dates)} to {max(dates)}")
        
        # Check if timestamps in interactions fall within this range
        if 'ts' in interactions.columns:
            interaction_times = pd.to_datetime(interactions['ts'], unit='s')
            print(f"Time range of interactions: {min(interaction_times)} to {max(interaction_times)}")
            
            # Check if ranges overlap
            features_min, features_max = min(dates), max(dates)
            interactions_min, interactions_max = min(interaction_times), max(interaction_times)
            
            if features_min <= interactions_max and features_max >= interactions_min:
                print("✅ Time ranges overlap correctly")
            else:
                print("❌ Warning: Time ranges do not overlap")
    
    # NEW: Check temporal consistency for sample users
    print("\nChecking temporal consistency for sample users...")
    
    # Convert interaction timestamps to dates
    interactions['date'] = pd.to_datetime(interactions['ts'], unit='s').dt.date
    
    # Convert user_features timestamps to dates
    feature_dates = {datetime.fromtimestamp(ts).date(): users for ts, users in user_features.items()}
    
    # Sample users to check (up to 10)
    sample_size = min(10, len(common_users))
    sample_users = random.sample(list(common_users), sample_size)
    
    for user_id in sample_users:
        # Count unique interaction days
        user_interaction_days = interactions[interactions['u'] == user_id]['date'].nunique()
        
        # Count days with embeddings
        user_embedding_days = sum(1 for date, users in feature_dates.items() if user_id in users)
        
        # Calculate ratio
        if user_interaction_days > 0:
            ratio = user_embedding_days / user_interaction_days
        else:
            ratio = 0
        
        status = "✅" if 0.9 <= ratio <= 1.1 else "❌"
        
        print(f"{status} User {user_id}: {user_embedding_days} embedding days / {user_interaction_days} interaction days ({ratio:.2f})")
    
    print("\nSummary:")
    if len(common_users) / len(interaction_user_ids) > 0.9:
        print("✅ Most interaction users have dynamic features")
    else:
        print("❌ Many interaction users are missing dynamic features")
    
    if len(missing_users) == 0:
        print("✅ All interaction users have dynamic features")
    else:
        print("❌ Some interaction users are missing dynamic features")
    
    print("\nDone with sanity check!")

if __name__ == "__main__":
    sanity_check_mappings()
