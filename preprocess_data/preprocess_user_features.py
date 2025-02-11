import duckdb
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from factorize import factorize
from tqdm import tqdm
import pickle
import os
import json
import numpy as np
import scipy.sparse as sp
import hashlib
from datetime import datetime, timedelta


def get_mapping_hash(mapping):
    """
    Create a deterministic hash of a mapping dictionary.
    """
    # Convert mapping to a sorted list of tuples to ensure consistent ordering
    sorted_items = sorted(mapping.items())
    # Convert to string and encode to bytes
    mapping_str = json.dumps(sorted_items)
    return hashlib.sha256(mapping_str.encode()).hexdigest()

def load_mapping(mapping_file):
    """
    Load a mapping from a JSON file. If the file doesn't exist, return an empty dict.
    """
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
    else:
        mapping = {}
    return mapping

def update_mapping(mapping, new_items):
    """
    Update the mapping with new items. New items are appended by assigning 
    them an index equal to the current length of the mapping.
    """
    for item in new_items:
        if item not in mapping:
            mapping[item] = len(mapping)
    return mapping


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
con = duckdb.connect('/home/sgan/scan_results.duckdb')
torch.manual_seed(42)  # IMPORTANT: temporary solution for deterministic results. Need this so that consumer_embeddings stays the same across runs.

def create_user_embedding(end_date):
    train_producer_df = con.execute(f"""
    WITH producers AS (
        SELECT 
            json_extract_string(record, '$.subject') as producer_did
        FROM records 
        WHERE collection = 'app.bsky.graph.follow'
        AND createdAt < '{end_date}'  -- before training cutoff date
        AND createdAt >= '2023-01-01' 
        GROUP BY json_extract_string(record, '$.subject')
        HAVING COUNT(*) >= 30
    )
    SELECT producer_did
    FROM producers
    """).fetchdf()

    # Get the edges (consumer-producer relationships)
    train_edges_df = con.execute("""
    SELECT 
        repo as consumer_did,
        json_extract_string(record, '$.subject') as producer_did
    FROM records
    WHERE 
        collection = 'app.bsky.graph.follow'
        AND json_extract_string(record, '$.subject') IN (SELECT producer_did FROM train_producer_df)
    """).fetchdf()

    
    # File paths for persistent mappings
    consumer_mapping_file = 'consumer_mapping.json'
    producer_mapping_file = 'producer_mapping.json'
    hash_file = 'mappings_hash.json'
    
    # Load existing mappings (or create new ones if they don't exist)
    consumer_to_idx = load_mapping(consumer_mapping_file)
    producer_to_idx = load_mapping(producer_mapping_file)
    
    # Store original hashes
    original_hashes = {
        'consumer': get_mapping_hash(consumer_to_idx),
        'producer': get_mapping_hash(producer_to_idx)
    }
    
    # Get new DIDs from the current training data
    new_consumers = train_edges_df['consumer_did'].unique().tolist()
    new_producers = train_producer_df['producer_did'].unique().tolist()
    
    # Update the mappings with any new DIDs
    consumer_to_idx = update_mapping(consumer_to_idx, new_consumers)
    producer_to_idx = update_mapping(producer_to_idx, new_producers)
    
    # Get new hashes
    new_hashes = {
        'consumer': get_mapping_hash(consumer_to_idx),
        'producer': get_mapping_hash(producer_to_idx)
    }
    
    # Check if mappings changed
    mappings_changed = (original_hashes != new_hashes)
    
    if mappings_changed:
        print("Warning: Mappings have changed! You should recompute post embeddings.")
        # Save the updated mappings to disk
        with open(consumer_mapping_file, 'w') as f:
            json.dump(consumer_to_idx, f)
        with open(producer_mapping_file, 'w') as f:
            json.dump(producer_to_idx, f)
        # Save the new hashes
        with open(hash_file, 'w') as f:
            json.dump(new_hashes, f)
    else:
        print("Mappings unchanged, safe to use existing post embeddings.")

    # Create sparse matrix in COO format; each edge has weight 1
    rows = [consumer_to_idx[consumer] for consumer in train_edges_df['consumer_did']]
    cols = [producer_to_idx[producer] for producer in train_edges_df['producer_did']]
    data = np.ones(len(rows))
    
    # Build the sparse matrix (then convert to CSR format for efficient multiplication)
    matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(len(consumer_to_idx), len(producer_to_idx))
    )
    
    print("Matrix shape:", matrix.shape)

    producer_community_affinities, consumer_embeddings, kmeans_cluster_centers = factorize(
        matrix, 
        algorithm='svd',
        n_components=64,
        n_clusters=100,
        device=device
    )

    return consumer_embeddings, consumer_to_idx


user_dynamic_features = {}
# Define start and end dates
start_date = datetime.strptime("2023-03-15", "%Y-%m-%d")
end_date = datetime.strptime("2023-06-30", "%Y-%m-%d")
embedding_dim=64

# Iterate through each day in the range
current_date = start_date
while current_date <= end_date:
    date_str = current_date.strftime("%Y-%m-%d")
    date_int = int(current_date.timestamp())

    # Get embeddings and consumer ID mapping
    consumer_embeddings, consumer_to_idx = create_user_embedding(date_str)

    # Query likes data for the given day
    likes_df = con.execute(f"""
        SELECT DISTINCT repo AS userId
        FROM records
        WHERE createdAt >= '{date_str}' 
            AND createdAt < '{(current_date + timedelta(days=1)).strftime("%Y-%m-%d")}'
            AND collection = 'app.bsky.feed.like'
    """).fetchdf()

    # Initialize dictionary for the current date
    user_dynamic_features[date_int] = {}

    # Process users
    for _, row in likes_df.iterrows():
        try:
            user_dynamic_features[date_int][row['userId']] = consumer_embeddings[consumer_to_idx[row['userId']]]
        except KeyError:  # If user is not found, assign zero vector
            user_dynamic_features[date_int][row['userId']] = np.zeros(embedding_dim)

    # Move to the next day
    current_date += timedelta(days=1)

print("Finished processing all dates.")

save_path = "/home/sgan/private/DyGLib/DG_data/bluesky/user_dynamic_features.pkl"

with open(save_path, "wb") as f:
    pickle.dump(user_dynamic_features, f)


user_mapping_path = "/home/sgan/private/DyGLib/DG_data/bluesky/user_mapping.pkl"

with open(user_mapping_path, "rb") as file:
    user_mapping = pickle.load(file)

# Replace user IDs with their mapped indices
user_dynamic_features_mapped = {
    date: {user_mapping[user]: emb for user, emb in users.items()} 
    for date, users in user_dynamic_features.items()
}

print("User IDs in user_dynamic_features have been replaced with user_mapping indices.")


save_path = "/home/sgan/private/DyGLib/DG_data/bluesky/user_dynamic_features.pkl"

with open(save_path, "wb") as f:
    pickle.dump(user_dynamic_features_mapped, f)