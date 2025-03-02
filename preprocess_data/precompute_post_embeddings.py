# Import required libraries
import pickle  
import os      
import numpy as np  
from datetime import datetime, timedelta  
import pandas as pd  
from tqdm import tqdm  
import logging  
import time
import duckdb
from dotenv import load_dotenv

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("../.env.local")
DUCKDB_PATH = os.getenv('DUCKDB_PATH')
DATA_PATH = "../DG_data/bluesky"
PROCESSED_DATA_PATH = "../processed_data/bluesky"

# Record start time for performance tracking
start_time = time.time()

# Load processed interaction data (already mapped to correct IDs)
logger.info("Loading processed interaction data...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'ml_bluesky.csv'))

# Convert timestamps to datetime objects
df['timestamp'] = pd.to_datetime(df['ts'], unit='s')

# Load original mappings
logger.info("Loading original mappings...")
with open(os.path.join(DATA_PATH, 'post_mapping.pkl'), 'rb') as f:
    post_mapping = pickle.load(f)
with open(os.path.join(DATA_PATH, 'user_mapping.pkl'), 'rb') as f:
    user_mapping = pickle.load(f)

# Calculate the offset used for post reindexing
num_users = len(user_mapping)
logger.info(f"Number of users: {num_users}")
logger.info(f"Sample post IDs in processed data: {sorted(df['i'].unique())[:5]}")

# Determine how post IDs were reindexed in preprocessing
# In preprocessing, posts get reindexed as: original_idx + user_count + 1
post_id_offset = num_users + 1
logger.info(f"Post ID offset: {post_id_offset}")

# Connect to DuckDB
logger.info("Connecting to DuckDB...")
con = duckdb.connect(DUCKDB_PATH)

# Get post creator information from the database
logger.info("Querying post creator information...")
post_creators_df = con.execute("""
    SELECT 
        repo AS creator_did, 
        repo || '_' || rkey AS post_key,
        createdAt AS created_at
    FROM records
    WHERE createdAt >= '2023-01-01' AND collection == 'app.bsky.feed.post'
""").fetchdf()

# Convert created_at to datetime
post_creators_df['created_at'] = pd.to_datetime(post_creators_df['created_at'])

# Map post keys to their original numeric IDs
post_creators_df['original_post_id'] = post_creators_df['post_key'].map(post_mapping)
post_creators_df = post_creators_df.dropna(subset=['original_post_id'])
post_creators_df['original_post_id'] = post_creators_df['original_post_id'].astype(int)

# Map to processed post IDs by applying the offset
post_creators_df['processed_post_id'] = post_creators_df['original_post_id'] + post_id_offset

# Map creator DIDs to their numeric IDs (with +1 since IDs start from 1)
post_creators_df['creator_id'] = post_creators_df['creator_did'].map(user_mapping) + 1
post_creators_df = post_creators_df.dropna(subset=['creator_id'])
post_creators_df['creator_id'] = post_creators_df['creator_id'].astype(int)

# Create a dictionary for easier lookup (mapping post ID to tuple of creator ID and creation date)
post_creators = dict(zip(post_creators_df['processed_post_id'], 
                         zip(post_creators_df['creator_id'], post_creators_df['created_at'])))

# Load mapped user dynamic features
logger.info("Loading user dynamic features...")
with open(os.path.join(PROCESSED_DATA_PATH, 'user_dynamic_features.pkl'), 'rb') as f:
    user_dynamic_features = pickle.load(f)

# Load producer features for initial post embeddings
logger.info("Loading producer dynamic features...")
with open(os.path.join(PROCESSED_DATA_PATH, 'producer_dynamic_features.pkl'), 'rb') as f:
    producer_dynamic_features = pickle.load(f)

# Convert date keys to datetime for easier lookup
date_to_timestamp = {datetime.fromtimestamp(ts).date(): ts for ts in user_dynamic_features.keys()}

# Add embedding date column (day of interaction) for temporal alignment
df['embedding_date'] = df['timestamp'].dt.date

# Prepare data for processing
logger.info("Preparing data...")
df = df.sort_values(['i', 'timestamp'])  # Sort by post and time
df = df[(df['timestamp'] >= '2023-03-15') & (df['timestamp'] <= '2023-03-22')]  # Filter by time range
grouped_posts = df.groupby('i')  # Group by post ID (already mapped)

# Initialize list to store embedding results
all_embeddings = []

# Stats tracking
missing_user_count = 0
processed_posts = 0
skipped_posts = 0
initial_embeddings_count = 0
missing_creator_count = 0

# First, add initial post embeddings where possible
logger.info("Adding initial post embeddings...")
embedding_dim = 64  # Adjust based on your actual embedding dimension

# Get all unique post IDs from the interaction data
all_post_ids = df['i'].unique()

for post_id in tqdm(all_post_ids):
    # Find the creator for this post
    creator_info = post_creators.get(post_id)
    
    if creator_info is None:
        missing_creator_count += 1
        continue
    
    # Unpack creator ID and creation date
    creator_id, created_at = creator_info
    creation_date = created_at.date()  # Get just the date part
    
    # Find the closest date in dynamic features
    if creation_date in date_to_timestamp:
        date_timestamp = date_to_timestamp[creation_date]
        
        # Track embedding source
        embedding_source = "zero"
        
        # Try to get the producer embedding first
        if creator_id in producer_dynamic_features.get(date_timestamp, {}):
            initial_embedding = producer_dynamic_features[date_timestamp][creator_id]
            embedding_source = "producer"
        # Fall back to consumer embedding
        elif creator_id in user_dynamic_features.get(date_timestamp, {}):
            initial_embedding = user_dynamic_features[date_timestamp][creator_id]
            embedding_source = "consumer"
        else:
            # If no embedding found, use zeros
            initial_embedding = np.zeros(embedding_dim, dtype=np.float32)
            embedding_source = "zero"
            
        all_embeddings.append({
            'post_id': int(post_id),
            'user_id': creator_id,  # Creator user ID
            'timestamp': created_at,  # Use actual creation timestamp
            'embedding': initial_embedding.astype(np.float32),
            'num_interactions': 0,  # 0 indicates initial embedding
            'embedding_source': embedding_source  # Add source information
        })
        initial_embeddings_count += 1

logger.info(f"Added {initial_embeddings_count} initial post embeddings")
logger.info(f"Missing creators for {missing_creator_count} posts")

# Process each post's interactions
logger.info("Processing post interactions...")
for post_id, post_interactions in tqdm(grouped_posts):
    try:
        # Get all interactions
        for i, row in post_interactions.iterrows():
            try:
                user_id = int(row['u'])  # User ID from processed data
                interaction_date = row['embedding_date']
                
                # Find the closest date in user_dynamic_features
                if interaction_date in date_to_timestamp:
                    date_timestamp = date_to_timestamp[interaction_date]
                    
                    if user_id in user_dynamic_features[date_timestamp]:
                        # Get user embedding for this interaction
                        user_embedding = user_dynamic_features[date_timestamp][user_id]
                        
                        all_embeddings.append({
                            'post_id': int(post_id),
                            'user_id': user_id,
                            'timestamp': row['timestamp'],
                            'embedding': user_embedding.astype(np.float32),
                            'num_interactions': 1,  # Single interaction
                            'embedding_source': 'consumer'  # All interactions use consumer embeddings
                        })
                    else:
                        missing_user_count += 1
            except Exception as e:
                logger.error(f"Error processing interaction for post {post_id}, user {row['u']}: {str(e)}")
                continue
        
        processed_posts += 1
                        
    except Exception as e:
        logger.error(f"Error processing post {post_id}: {str(e)}")
        continue

logger.info(f"Processed {processed_posts} posts with {missing_user_count} missing users")
logger.info(f"Created {initial_embeddings_count} initial embeddings and {len(all_embeddings) - initial_embeddings_count} interaction embeddings")
logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")

# Save as parquet
post_embeddings_df = pd.DataFrame(all_embeddings)
post_embeddings_df = post_embeddings_df.sort_values('timestamp')
output_file = os.path.join(os.path.expanduser("~"), 'post_dynamic_embeddings.parquet')
post_embeddings_df.to_parquet(output_file, compression='snappy')

# Verify embeddings
logger.info("Loading embeddings to verify...")
post_embeddings_path = os.path.join(os.path.expanduser("~"), 'post_dynamic_embeddings.parquet')
post_embeddings = pd.read_parquet(post_embeddings_path)
logger.info(f"Successfully loaded {len(post_embeddings)} post embeddings")