import duckdb
import pandas as pd
from typing import Tuple
from torch_geometric.data import Data

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

    con.close()
    
    return likes_df, follows_df, posts_df