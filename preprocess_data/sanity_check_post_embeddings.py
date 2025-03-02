import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
from collections import Counter, defaultdict
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
EMBEDDINGS_PATH = os.path.join(os.path.expanduser("~"), 'post_dynamic_embeddings.parquet')
PROCESSED_DATA_PATH = "../processed_data/bluesky"
DATA_PATH = "../DG_data/bluesky"

def load_data():
    """Load embeddings and related data"""
    logger.info(f"Loading post embeddings from {EMBEDDINGS_PATH}")
    embeddings_df = pd.read_parquet(EMBEDDINGS_PATH)
    
    logger.info("Loading original interaction data")
    interactions_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, 'ml_bluesky.csv'))
    interactions_df['timestamp'] = pd.to_datetime(interactions_df['ts'], unit='s')
    
    logger.info("Loading user and post mappings")
    with open(os.path.join(DATA_PATH, 'user_mapping.pkl'), 'rb') as f:
        user_mapping = pickle.load(f)
    
    with open(os.path.join(DATA_PATH, 'post_mapping.pkl'), 'rb') as f:
        post_mapping = pickle.load(f)
    
    return embeddings_df, interactions_df, user_mapping, post_mapping

def basic_statistics(embeddings_df, interactions_df):
    """Compute and display basic statistics about the data"""
    logger.info("Computing basic statistics")
    
    # Post embedding stats
    num_posts = embeddings_df['post_id'].nunique()
    num_total_embeddings = len(embeddings_df)
    num_initial_embeddings = len(embeddings_df[embeddings_df['num_interactions'] == 0])
    num_interaction_embeddings = len(embeddings_df[embeddings_df['num_interactions'] > 0])
    
    # Embedding sources
    embedding_sources = embeddings_df['embedding_source'].value_counts().to_dict()
    
    # Temporal distribution
    time_range = (embeddings_df['timestamp'].min(), embeddings_df['timestamp'].max())
    time_span = time_range[1] - time_range[0]
    
    # Interactions per post
    interactions_per_post = embeddings_df.groupby('post_id').size().describe()
    
    # Print statistics
    logger.info(f"Number of unique posts: {num_posts}")
    logger.info(f"Number of total embeddings: {num_total_embeddings}")
    logger.info(f"Number of initial embeddings: {num_initial_embeddings}")
    logger.info(f"Number of interaction embeddings: {num_interaction_embeddings}")
    logger.info(f"Embedding sources: {embedding_sources}")
    logger.info(f"Time range: {time_range[0]} to {time_range[1]} ({time_span})")
    logger.info(f"Interactions per post statistics:\n{interactions_per_post}")
    
    return {
        'num_posts': num_posts,
        'num_total_embeddings': num_total_embeddings,
        'num_initial_embeddings': num_initial_embeddings,
        'num_interaction_embeddings': num_interaction_embeddings,
        'embedding_sources': embedding_sources,
        'time_range': time_range,
        'interactions_per_post': interactions_per_post
    }

def check_consistency(embeddings_df):
    """Check for data consistency issues"""
    logger.info("Checking data consistency")
    
    # Check for duplicates (excluding the embedding column)
    duplicate_rows = embeddings_df.drop(columns=['embedding']).duplicated().sum()
    logger.info(f"Number of duplicate rows (excluding embedding column): {duplicate_rows}")
    
    # Check for missing values
    missing_values = embeddings_df.isnull().sum()
    logger.info(f"Missing values:\n{missing_values}")
    
    # Check for chronological ordering within each post
    chronological_errors = 0
    for post_id, group in tqdm(embeddings_df.groupby('post_id')):
        sorted_timestamps = group['timestamp'].sort_values()
        if not sorted_timestamps.equals(group['timestamp']):
            chronological_errors += 1
    
    logger.info(f"Posts with chronological ordering errors: {chronological_errors}")
    
    # Verify interaction counts make sense
    max_interactions = embeddings_df['num_interactions'].max()
    logger.info(f"Maximum interactions for a post: {max_interactions}")
    
    # Check embedding dimensions
    sample_embedding = embeddings_df.iloc[0]['embedding']
    embedding_dim = len(sample_embedding)
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Check for unusual values in embeddings
    nan_embeddings = 0
    zero_embeddings = 0
    for idx, row in tqdm(embeddings_df.iterrows(), total=len(embeddings_df)):
        if np.isnan(row['embedding']).any():
            nan_embeddings += 1
        if np.all(row['embedding'] == 0):
            zero_embeddings += 1
    
    logger.info(f"Embeddings with NaN values: {nan_embeddings}")
    logger.info(f"Embeddings that are all zeros: {zero_embeddings}")
    
    return {
        'duplicate_rows': duplicate_rows,
        'missing_values': missing_values,
        'chronological_errors': chronological_errors,
        'max_interactions': max_interactions,
        'embedding_dim': embedding_dim,
        'nan_embeddings': nan_embeddings,
        'zero_embeddings': zero_embeddings
    }

def check_cumulative_averaging(embeddings_df):
    """Verify that the cumulative averaging is working correctly"""
    logger.info("Checking cumulative averaging logic")
    
    # Sample a few posts to check in detail
    sample_posts = embeddings_df['post_id'].sample(min(5, embeddings_df['post_id'].nunique()))
    
    for post_id in sample_posts:
        post_embeddings = embeddings_df[embeddings_df['post_id'] == post_id].sort_values('timestamp')
        
        logger.info(f"\nExamining post {post_id}:")
        logger.info(f"  Number of embeddings: {len(post_embeddings)}")
        
        # Check if we have an initial embedding
        has_initial = (post_embeddings['num_interactions'] == 0).any()
        logger.info(f"  Has initial embedding: {has_initial}")
        
        if has_initial:
            initial_embedding = post_embeddings[post_embeddings['num_interactions'] == 0]['embedding'].iloc[0]
            initial_source = post_embeddings[post_embeddings['num_interactions'] == 0]['embedding_source'].iloc[0]
            logger.info(f"  Initial embedding source: {initial_source}")
            
            # Verify the num_interactions increases correctly
            interactions = post_embeddings[post_embeddings['num_interactions'] > 0]
            if len(interactions) > 0:
                expected_sequence = list(range(1, len(interactions) + 1))
                actual_sequence = interactions['num_interactions'].tolist()
                is_sequence_correct = expected_sequence == actual_sequence
                logger.info(f"  Interaction count sequence correct: {is_sequence_correct}")
                if not is_sequence_correct:
                    logger.warning(f"  Expected: {expected_sequence}, Got: {actual_sequence}")
        
        # For the first few posts, also check if the embedding changes are reasonable
        if post_id == sample_posts.iloc[0]:
            logger.info("  Detailed embedding analysis:")
            all_embeddings = [row['embedding'] for _, row in post_embeddings.iterrows()]
            
            # Calculate cosine similarity between consecutive embeddings
            if len(all_embeddings) > 1:
                similarities = []
                for i in range(1, len(all_embeddings)):
                    dot_product = np.dot(all_embeddings[i-1], all_embeddings[i])
                    norm1 = np.linalg.norm(all_embeddings[i-1])
                    norm2 = np.linalg.norm(all_embeddings[i])
                    similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
                    similarities.append(similarity)
                
                logger.info(f"  Cosine similarities between consecutive embeddings: {similarities}")
                logger.info(f"  Average similarity: {np.mean(similarities)}")

def visualize_embeddings(embeddings_df, stats):
    """Create visualizations to understand the data better"""
    logger.info("Creating visualizations")
    
    # Set up the figure
    plt.figure(figsize=(15, 10))
    
    # 1. Histogram of interactions per post
    plt.subplot(2, 2, 1)
    interactions_per_post = embeddings_df.groupby('post_id').size()
    sns.histplot(interactions_per_post, kde=True)
    plt.title('Distribution of Interactions per Post')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Count')
    
    # 2. Pie chart of embedding sources
    plt.subplot(2, 2, 2)
    source_counts = embeddings_df['embedding_source'].value_counts()
    plt.pie(source_counts, labels=source_counts.index, autopct='%1.1f%%')
    plt.title('Embedding Sources')
    
    # 3. Activity over time
    plt.subplot(2, 2, 3)
    embeddings_df['date'] = embeddings_df['timestamp'].dt.date
    daily_activity = embeddings_df.groupby('date').size()
    daily_activity.plot(kind='line')
    plt.title('Activity Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Embeddings')
    
    # 4. Embedding norms distribution
    plt.subplot(2, 2, 4)
    embedding_norms = [np.linalg.norm(row['embedding']) for _, row in tqdm(embeddings_df.iterrows())]
    sns.histplot(embedding_norms, kde=True)
    plt.title('Distribution of Embedding Norms')
    plt.xlabel('Norm Value')
    plt.ylabel('Count')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('embedding_analysis.png')
    logger.info("Visualizations saved to embedding_analysis.png")

def main():
    # Load the data
    embeddings_df, interactions_df, user_mapping, post_mapping = load_data()
    
    # Perform basic statistics
    stats = basic_statistics(embeddings_df, interactions_df)
    
    # Check for consistency issues
    consistency = check_consistency(embeddings_df)
    
    # Check cumulative averaging
    check_cumulative_averaging(embeddings_df)
    
    # Create visualizations
    visualize_embeddings(embeddings_df, stats)
    
    logger.info("Sanity check completed successfully")

if __name__ == "__main__":
    main()