# Import required libraries
import pickle  
import os      
import numpy as np  
from datetime import datetime, timedelta  
import pandas as pd  
from tqdm import tqdm  
import logging  
import time  

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths for data
PROCESSED_DATA_PATH = "../processed_data/bluesky"

# Record start time for performance tracking
start_time = time.time()

# Load processed interaction data (already mapped to correct IDs)
logger.info("Loading processed interaction data...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'ml_bluesky.csv'))

# Convert timestamps to datetime objects
df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
# df = df[(df['timestamp'] >= '2023-06-01')]

# Load mapped user dynamic features
logger.info("Loading user dynamic features...")
with open(os.path.join(PROCESSED_DATA_PATH, 'user_dynamic_features.pkl'), 'rb') as f:
    user_dynamic_features = pickle.load(f)

# Convert date keys to datetime for easier lookup
date_to_timestamp = {datetime.fromtimestamp(ts).date(): ts for ts in user_dynamic_features.keys()}

# Add embedding date column (day of interaction) for temporal alignment
df['embedding_date'] = df['timestamp'].dt.date

# Prepare data for processing
logger.info("Preparing data...")
df = df.sort_values(['i', 'timestamp'])  # Sort by post and time
grouped_posts = df.groupby('i')  # Group by post ID (already mapped)

# Initialize list to store embedding results
all_embeddings = []

# Stats tracking
missing_user_count = 0
processed_posts = 0
skipped_posts = 0

# Process each post's interactions
logger.info("Processing posts...")
for post_id, post_interactions in tqdm(grouped_posts):
    try:
        # OPTION 1: Get all interactions (not just first 24 hours)
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
                            'user_id': user_id,  # Added user_id to the data
                            'timestamp': row['timestamp'],
                            'embedding': user_embedding.astype(np.float32),
                            'num_interactions': 1  # Single interaction
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
logger.info(f"Processing completed in {time.time() - start_time:.2f} seconds")

# Save as parquet
post_embeddings_df = pd.DataFrame(all_embeddings)
output_file = os.path.join(os.path.expanduser("~"), 'post_dynamic_embeddings.parquet')
post_embeddings_df.to_parquet(output_file, compression='snappy')

# Verify embeddings
logger.info("Loading embeddings to verify...")
post_embeddings_path = os.path.join(os.path.expanduser("~"), 'post_dynamic_embeddings.parquet')
post_embeddings = pd.read_parquet(post_embeddings_path)
logger.info(f"Successfully loaded {len(post_embeddings)} post embeddings")