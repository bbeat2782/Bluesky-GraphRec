import duckdb
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from typing import Tuple, Dict

def load_interactions() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load interaction and follow data from DuckDB before May 2023 using URI format."""
    con = duckdb.connect('../random_tests/scan_results.duckdb')
    
    # Get likes with URI format
    likes_df = con.execute("""
        SELECT 
            'at://' || repo || '/app.bsky.feed.like/' || rkey as interaction_uri,
            json_extract_string(record, '$.subject.uri') as post_uri,
            repo as user_uri,
            createdAt as timestamp
        FROM records 
        WHERE collection = 'app.bsky.feed.like'
            AND createdAt < '2023-05-01'
    """).fetchdf()
    
    # Get follows
    follows_df = con.execute("""
        SELECT 
            'at://' || repo || '/app.bsky.graph.follow/' || rkey as follow_uri,
            repo as follower_uri,
            json_extract_string(record, '$.subject') as following_uri,
            createdAt as timestamp
        FROM records 
        WHERE collection = 'app.bsky.graph.follow'
            AND createdAt < '2023-05-01'
    """).fetchdf()
    
    # Get posts that were liked
    posts_df = con.execute("""
        SELECT DISTINCT
            json_extract_string(record, '$.subject.uri') as post_uri,
            createdAt
        FROM records
        WHERE collection = 'app.bsky.feed.like'
            AND createdAt < '2023-05-01'
    """).fetchdf()
    
    # Remove any rows with NULL values
    likes_df = likes_df.dropna()
    follows_df = follows_df.dropna()
    posts_df = posts_df.dropna()
    
    print(f"Loaded {len(likes_df)} likes, {len(follows_df)} follows, and {len(posts_df)} unique posts before May 2023")
    
    return likes_df, follows_df, posts_df

def create_interaction_matrices(likes_df: pd.DataFrame, follows_df: pd.DataFrame) -> Tuple[csr_matrix, csr_matrix, Dict, Dict]:
    """Create interaction matrices and ID mappings using URI format."""
    # Create unified user mapping from both likes and follows
    all_users = pd.concat([
        likes_df['user_uri'],
        follows_df['follower_uri'],
        follows_df['following_uri']
    ]).unique()
    
    user_mapping = {uid: idx for idx, uid in enumerate(all_users)}
    post_mapping = {pid: idx for idx, pid in enumerate(likes_df['post_uri'].unique())}
    
    # Create user-post interaction matrix
    row = [user_mapping[u] for u in likes_df['user_uri']]
    col = [post_mapping[p] for p in likes_df['post_uri']]
    data = np.ones(len(likes_df))
    
    interaction_matrix = csr_matrix(
        (data, (row, col)), 
        shape=(len(user_mapping), len(post_mapping))
    )
    
    # Create user-user follow matrix
    row_f = [user_mapping[u] for u in follows_df['follower_uri']]
    col_f = [user_mapping[u] for u in follows_df['following_uri']]
    data_f = np.ones(len(follows_df))
    
    follow_matrix = csr_matrix(
        (data_f, (row_f, col_f)), 
        shape=(len(user_mapping), len(user_mapping))
    )
    
    return interaction_matrix, follow_matrix, user_mapping, post_mapping

def create_adj_matrix(interaction_matrix: csr_matrix, follow_matrix: csr_matrix) -> torch.sparse.FloatTensor:
    """Create normalized adjacency matrix for the heterogeneous graph using numerically stable normalization."""
    n_users, n_items = interaction_matrix.shape
    
    # Create adjacency matrix [[0, R], [R.T, 0]]
    adj = sp.vstack([
        sp.hstack([sp.csr_matrix((n_users, n_users)), interaction_matrix]),
        sp.hstack([interaction_matrix.transpose(), sp.csr_matrix((n_items, n_items))])
    ])
    
    # Add self-loops
    adj = adj + sp.eye(adj.shape[0])
    
    # Calculate degree matrix
    rowsum = np.array(adj.sum(1))
    
    # Calculate D^(-1/2)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # Handle division by zero
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    
    # Create diagonal matrix
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Calculate normalized adjacency: D^(-1/2) A D^(-1/2)
    adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    # Convert to COO format for PyTorch
    adj = adj.tocoo()
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    
    # Convert to PyTorch sparse tensor
    adj = torch.sparse.FloatTensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(adj.shape)
    )
    
    return adj