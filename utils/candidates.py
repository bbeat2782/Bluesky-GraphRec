import numpy as np
import pandas as pd
import pickle
import os
from datetime import timedelta
from sklearn.metrics.pairwise import cosine_similarity
import logging

class EmbeddingCandidateEdgeSampler:
    """
    Candidate edge sampler that uses embedding similarity for candidate generation.
    """
    def __init__(self, user_dynamic_features, post_embeddings, original_df=None, time_window_hours=1, 
                 n_candidates=100, seed=None):
        """
        Initialize the embedding-based candidate sampler.
        
        Args:
            user_dynamic_features: Dictionary or DataFrame of user embeddings
            post_embeddings: List of dictionaries with post embeddings
            original_df: Original interaction DataFrame (for filtering)
            time_window_hours: Hours to look back for post candidates
            n_candidates: Number of candidates to return
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.user_dynamic_features = user_dynamic_features
        
        # Convert post embeddings to DataFrame if it's a list
        if isinstance(post_embeddings, list):
            self.post_embeddings_df = pd.DataFrame(post_embeddings)
        else:
            self.post_embeddings_df = post_embeddings
            
        self.original_df = original_df
        self.time_window_hours = time_window_hours
        self.n_candidates = n_candidates
        self.seed = seed
        
        if isinstance(self.user_dynamic_features, dict):
            # Convert user features dictionary to DataFrame for easier manipulation
            self.user_dynamic_features_df = pd.DataFrame.from_dict(self.user_dynamic_features, orient='index')
            self.user_dynamic_features_df.index = pd.to_datetime(self.user_dynamic_features_df.index, unit='s')
            self.user_dynamic_features_df = self.user_dynamic_features_df.sort_index()
        else:
            self.user_dynamic_features_df = self.user_dynamic_features
            
        self.logger.info(f"Initialized EmbeddingCandidateEdgeSampler with {len(self.post_embeddings_df)} post embeddings")
        
        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def reset_random_state(self):
        """Reset random state for reproducibility during evaluation"""
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def sample(self, size, batch_src_node_ids=None, batch_dst_node_ids=None, batch_node_interact_times=None, 
               current_batch_start_time=None, popularity_based=False):
        """
        Sample candidate edges for each interaction.
        
        Args:
            size: Number of interactions to sample for
            batch_src_node_ids: Source node IDs (users)
            batch_dst_node_ids: Destination node IDs (posts that users interacted with)
            batch_node_interact_times: Timestamps of interactions
            current_batch_start_time: Not used, kept for compatibility
            popularity_based: Whether to use popularity-based sampling (fallback)
        
        Returns:
            Dictionary mapping interaction times to candidate post IDs
        """
        candidates_dict = {}
        
        # Process each interaction
        for i in range(size):
            user_id = batch_src_node_ids[i]
            timestamp = pd.Timestamp(batch_node_interact_times[i], unit='s')
            true_post_id = batch_dst_node_ids[i]
            
            # Get embedding date (7am of the day)
            embedding_date = pd.Timestamp(timestamp.date()) + pd.Timedelta(hours=7)
            
            # Get user embedding
            try:
                if embedding_date in self.user_dynamic_features_df.index and user_id in self.user_dynamic_features_df.columns:
                    user_embedding = self.user_dynamic_features_df.loc[embedding_date, user_id]
                    
                    # Skip if user embedding is not available
                    if not isinstance(user_embedding, np.ndarray):
                        # Fallback to random sampling
                        random_candidates = np.random.choice(
                            self.post_embeddings_df['post_id'].unique(), 
                            size=self.n_candidates, 
                            replace=False
                        )
                        candidates_dict[batch_node_interact_times[i]] = random_candidates
                        continue
                        
                    # Get posts active within time window
                    time_window_start = timestamp - timedelta(hours=self.time_window_hours)
                    active_posts = self.post_embeddings_df[
                        (self.post_embeddings_df['timestamp'] <= timestamp) & 
                        (self.post_embeddings_df['timestamp'] >= time_window_start)
                    ]
                    
                    if len(active_posts) == 0:
                        # Fallback to random sampling
                        random_candidates = np.random.choice(
                            self.post_embeddings_df['post_id'].unique(), 
                            size=self.n_candidates, 
                            replace=False
                        )
                        candidates_dict[batch_node_interact_times[i]] = random_candidates
                        continue
                    
                    # Get latest embedding for each post
                    latest_embeddings = (
                        active_posts.groupby('post_id')
                        .last()
                        .reset_index()
                    )
                    
                    # Calculate similarities
                    post_embeddings = np.stack(latest_embeddings['embedding'].values)
                    similarities = cosine_similarity([user_embedding], post_embeddings)[0]
                    
                    # Get top N candidates
                    top_indices = np.argsort(similarities)[-self.n_candidates:][::-1]
                    candidate_posts = latest_embeddings.iloc[top_indices]['post_id'].values
                    
                    # Make sure true post is in candidates for evaluation
                    if true_post_id not in candidate_posts:
                        # Replace the last candidate with the true post
                        candidate_posts[-1] = true_post_id
                        
                    candidates_dict[batch_node_interact_times[i]] = candidate_posts
                else:
                    # Fallback to random sampling
                    random_candidates = np.random.choice(
                        self.post_embeddings_df['post_id'].unique(), 
                        size=self.n_candidates, 
                        replace=False
                    )
                    candidates_dict[batch_node_interact_times[i]] = random_candidates
            except Exception as e:
                self.logger.error(f"Error generating candidates for user {user_id} at time {timestamp}: {str(e)}")
                # Fallback to random sampling
                random_candidates = np.random.choice(
                    self.post_embeddings_df['post_id'].unique(), 
                    size=self.n_candidates, 
                    replace=False
                )
                candidates_dict[batch_node_interact_times[i]] = random_candidates
                
        return candidates_dict