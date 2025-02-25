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
DATA_PATH = "../DG_data/bluesky"
PROCESSED_DATA_PATH = "../processed_data/bluesky"

# Record start time for performance tracking
start_time = time.time()

# Load and preprocess the main dataset
logger.info("Loading data...")
df = pd.read_csv(os.path.join(DATA_PATH, 'bluesky.csv'))
# Convert timestamp strings to datetime objects
df['timestamp'] = pd.to_datetime(df['timestamp'].astype(str), format='%Y%m%d%H%M%S')
# Filter data from March 15, 2023 onwards
df = df[df['timestamp'] >= '2023-03-15']
# df = df[(df['timestamp'] >= '2023-03-15') & (df['timestamp'] < '2023-03-22')]

# Load user dynamic features from pickle file
with open(os.path.join(DATA_PATH, 'user_dynamic_features.pkl'), 'rb') as f:
    user_dynamic_features = pickle.load(f)

# Convert user features dictionary to DataFrame for easier manipulation
user_dynamic_features_df = pd.DataFrame.from_dict(user_dynamic_features, orient='index')
user_dynamic_features_df.index = pd.to_datetime(user_dynamic_features_df.index, unit='s')
user_dynamic_features_df = user_dynamic_features_df.sort_index()

# Add embedding date column (7am of each day) for temporal alignment
df['embedding_date'] = df['timestamp'].dt.date.apply(
    lambda x: pd.Timestamp(x) + pd.Timedelta(hours=7)
)

# Prepare data for processing
logger.info("Preparing data...")
df = df.sort_values(['destination_node', 'timestamp'])  # Sort by post and time
grouped_posts = df.groupby('destination_node')  # Group by post ID

# Initialize list to store embedding results
all_embeddings = []

# Stats tracking
missing_user_count = 0

# Process each post's interactions
logger.info("Processing posts...")
for post_id, post_interactions in tqdm(grouped_posts): # post_id = destination_node, post_interactions = df. Basically: i,x
    """
    Example of grouped_posts structure:
    grouped_posts = {
    post_id_1: [
        {timestamp: t1, source_node: user1, ...},
        {timestamp: t2, source_node: user2, ...},
        ...
    ],
    post_id_2: [
        {timestamp: t3, source_node: user3, ...}, 
        {timestamp: t4, source_node: user4, ...},
        ...
    ],
    ...
    }
    """
    try:
        first_interaction = post_interactions['timestamp'].iloc[0]
        
        # Get all interactions within 24 hours of first interaction
        end_time = first_interaction + pd.Timedelta(hours=24)
        first_24h = post_interactions[
            (post_interactions['timestamp'] >= first_interaction) & 
            (post_interactions['timestamp'] <= end_time)
        ]
        
        if len(first_24h) == 0:
            continue
            
        # Process each interaction in chronological order
        valid_embeddings = []  # Just use a list instead of pre-allocating array
        
        for i, row in first_24h.iterrows():
            user_id = row['source_node']
            date = row['embedding_date']
            
            if user_id in user_dynamic_features_df.columns:
                embedding = user_dynamic_features_df.loc[date, user_id]
                # if isinstance(embedding, np.ndarray):
                valid_embeddings.append(embedding)
                # Calculate average of all embeddings so far
                avg_embedding = np.mean(valid_embeddings, axis=0)
                all_embeddings.append({
                    'post_id': post_id,
                    'timestamp': row['timestamp'],
                    'embedding': avg_embedding.astype(np.float32),
                    'num_interactions': len(valid_embeddings)
                })
            else:
                missing_user_count += 1
                        
    except Exception as e:
        logger.error(f"Error processing post {post_id}: {str(e)}")
        continue

# Save all embeddings directly
output_file = os.path.join(PROCESSED_DATA_PATH, 'post_dynamic_embeddings.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(all_embeddings, f)