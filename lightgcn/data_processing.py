import duckdb
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from typing import Tuple, Dict

def load_interactions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load interaction data from DuckDB before May 2023 using URI format."""
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
    
    # Get posts that were liked before May 2023
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
    posts_df = posts_df.dropna()
    
    print(f"Loaded {len(likes_df)} interactions and {len(posts_df)} unique posts before May 2023")
    
    return likes_df, posts_df

def create_interaction_matrix(likes_df: pd.DataFrame) -> Tuple[csr_matrix, Dict, Dict]:
    """Create interaction matrix and ID mappings using URI format."""
    # Create mappings
    user_mapping = {uid: idx for idx, uid in enumerate(likes_df['user_uri'].unique())}
    post_mapping = {pid: idx for idx, pid in enumerate(likes_df['post_uri'].unique())}
    
    # Convert to matrix format
    row = [user_mapping[u] for u in likes_df['user_uri']]
    col = [post_mapping[p] for p in likes_df['post_uri']]
    data = np.ones(len(likes_df))
    
    # Create sparse matrix
    interaction_matrix = csr_matrix(
        (data, (row, col)), 
        shape=(len(user_mapping), len(post_mapping))
    )
    
    return interaction_matrix, user_mapping, post_mapping

def create_adj_matrix(interaction_matrix: csr_matrix) -> torch.sparse.FloatTensor:
    """Create normalized adjacency matrix for the graph."""
    # Create adjacency matrix in COO format
    n_users, n_items = interaction_matrix.shape
    
    # Create adjacency matrix [[0, R], [R.T, 0]] where R is the interaction matrix
    adj = sp.vstack([
        sp.hstack([sp.csr_matrix((n_users, n_users)), interaction_matrix]),
        sp.hstack([interaction_matrix.transpose(), sp.csr_matrix((n_items, n_items))])
    ])
    
    # Convert to PyTorch sparse tensor
    adj = adj.tocoo()
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    
    # Normalize adjacency matrix
    adj = torch.sparse.FloatTensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        torch.Size(adj.shape)
    )
    
    return adj