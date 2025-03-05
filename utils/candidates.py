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
    def __init__(self, user_dynamic_features, post_embeddings_df, time_window_hours=24, 
                 n_candidates=1000, seed=None, include_true_dst=True):
        """
        Initialize the embedding-based candidate sampler.
        
        Args:
            user_dynamic_features: Dictionary of user embeddings
            post_embeddings_df: DataFrame with post embeddings
            time_window_hours: Hours to look back for post candidates
            n_candidates: Number of candidates to return
            seed: Random seed for reproducibility
            include_true_dst: Whether to include the true destination in candidates
        """
        self.logger = logging.getLogger(__name__)
        
        # Store the user dynamic features directly without adjustment
        self.user_dynamic_features = user_dynamic_features
        
        self.post_embeddings_df = post_embeddings_df
        self.time_window_hours = time_window_hours
        self.n_candidates = n_candidates
        self.seed = seed
        self.include_true_dst = include_true_dst
        
        # # Convert user_dynamic_features to DataFrame for easier access
        # self.user_dynamic_features_df = pd.DataFrame.from_dict(self.user_dynamic_features, orient='index')
        # self.user_dynamic_features_df.index = pd.to_datetime(self.user_dynamic_features_df.index, unit='s')
        # self.user_dynamic_features_df = self.user_dynamic_features_df.sort_index()
            
        self.logger.info(f"Initialized EmbeddingCandidateEdgeSampler with {len(self.post_embeddings_df)} post embeddings")
        
        # Set random seed if provided
        self.reset_random_state()
        
        # Cache for post embeddings by day to speed up retrieval
        self.post_embeddings_cache = {}
        
        # Debug counters
        self.true_post_added_count = 0
        self.total_processed = 0
        
        # Detailed fallback counters
        self.fallback_counters = {
            "user_embedding_not_available": 0,
            "embedding_date_not_found": 0,
            "user_id_not_found": 0,
            "no_active_posts": 0,
            "exception_occurred": 0
        }
        
        # Hit rate tracking
        self.hit_counters = {20: 0, 50: 0, 100: 0, 500: 0, 1000: 0, 2000: 0, 3000: 0, 5000: 0}
        self.k_values = sorted(self.hit_counters.keys())
    
    def reset_random_state(self):
        """Reset random state for reproducibility during evaluation"""
        if self.seed is not None:
            np.random.seed(self.seed)
    
    def sample(self, size, batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, 
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
        debug_info = []  # For debugging            
        
        # Process each interaction
        for i in range(size):
            self.total_processed += 1
            user_id = batch_src_node_ids[i]
            timestamp = pd.Timestamp(batch_node_interact_times[i], unit='s')
            true_post_id = batch_dst_node_ids[i]
            
            # Get embedding date (7am of the day)
            embedding_date = pd.Timestamp(timestamp.date()) + pd.Timedelta(hours=7)
            embedding_date_int = int(embedding_date.timestamp())
            
            # For debugging
            user_info = {
                "user_id": user_id,
                "timestamp": timestamp,
                "true_post_id": true_post_id,
                "embedding_date": embedding_date
            }
            
            # Get user embedding
            try:
                # Check if embedding date exists
                if embedding_date_int not in self.user_dynamic_features:
                    user_info["error"] = "Embedding date not found"
                    debug_info.append(user_info)
                    self.fallback_counters["embedding_date_not_found"] += 1
                    random_candidates = np.random.choice(
                        self.post_embeddings_df['post_id'].unique(), 
                        size=self.n_candidates, 
                        replace=False
                    )
                    candidates_dict[batch_node_interact_times[i]] = random_candidates
                    continue
                
                # Check if user ID exists in the date's dictionary
                if user_id not in self.user_dynamic_features[embedding_date_int]:
                    user_info["error"] = "User ID not found"
                    debug_info.append(user_info)
                    self.fallback_counters["user_id_not_found"] += 1
                    random_candidates = np.random.choice(
                        self.post_embeddings_df['post_id'].unique(), 
                        size=self.n_candidates, 
                        replace=False
                    )
                    candidates_dict[batch_node_interact_times[i]] = random_candidates
                    continue
                
                # Get user embedding directly from the nested dictionary
                user_embedding = self.user_dynamic_features[embedding_date_int][user_id]
                
                # # Print user embedding info
                # print("user_embedding type: ", type(user_embedding))
                # print("user_embedding: ", user_embedding)
                # print("user_info: ", user_info)

                
                # Skip if user embedding is not available
                if not isinstance(user_embedding, np.ndarray):
                    user_info["error"] = "User embedding not available"
                    debug_info.append(user_info)
                    self.fallback_counters["user_embedding_not_available"] += 1
                    random_candidates = np.random.choice(
                        self.post_embeddings_df['post_id'].unique(), 
                        size=self.n_candidates, 
                        replace=False
                    )
                    candidates_dict[batch_node_interact_times[i]] = random_candidates
                    continue
                    
                # Get posts active within time window
                time_window_start = timestamp - timedelta(hours=self.time_window_hours)
                
                # Use cache for post embeddings if available
                day_key = timestamp.date().isoformat()
                if day_key in self.post_embeddings_cache:
                    active_posts = self.post_embeddings_cache[day_key]
                else:
                    active_posts = self.post_embeddings_df[
                        (self.post_embeddings_df['timestamp'] < timestamp) & 
                        (self.post_embeddings_df['timestamp'] >= time_window_start)
                    ]
                    self.post_embeddings_cache[day_key] = active_posts
                
                user_info["num_active_posts"] = len(active_posts)
                
                if len(active_posts) == 0:
                    user_info["error"] = "No active posts in time window"
                    debug_info.append(user_info)
                    self.fallback_counters["no_active_posts"] += 1
                    random_candidates = np.random.choice(
                        self.post_embeddings_df['post_id'].unique(), 
                        size=self.n_candidates, 
                        replace=False
                    )
                    candidates_dict[batch_node_interact_times[i]] = random_candidates
                    continue
                
                # Get latest embedding for each post
                latest_embeddings = (
                    active_posts.sort_values('timestamp')
                    .groupby('post_id')
                    .last()
                    .reset_index()
                )
                
                # Calculate similarities
                post_embeddings = np.stack(latest_embeddings['embedding'].values)
                similarities = cosine_similarity([user_embedding], post_embeddings)[0]
                
                # Get top N candidates
                top_indices = np.argsort(similarities)[-self.n_candidates:][::-1]
                candidate_posts = latest_embeddings.iloc[top_indices]['post_id'].values
                top_similarities = similarities[top_indices]
                
                # Check if true post is in candidates and track it
                true_post_in_candidates = true_post_id in candidate_posts
                user_info["true_post_in_candidates"] = true_post_in_candidates
                
                # Find position of true post in the ranked list (if present)
                true_post_position = None
                for idx, post_id in enumerate(candidate_posts):
                    if post_id == true_post_id:
                        true_post_position = idx
                        break
                
                # Update hit counters for each k value
                if true_post_position is not None:
                    for k in self.k_values:
                        if true_post_position < k:
                            self.hit_counters[k] += 1
                
                # Make sure true post is in candidates for evaluation if needed
                if self.include_true_dst and not true_post_in_candidates:
                    # Replace the last candidate with the true post
                    candidate_posts[-1] = true_post_id
                    self.true_post_added_count += 1
                    user_info["true_post_added"] = True
                
                user_info["top_similarity"] = float(top_similarities[0]) if len(top_similarities) > 0 else None
                debug_info.append(user_info)
                    
                candidates_dict[batch_node_interact_times[i]] = candidate_posts
                
            except Exception as e:
                user_info["error"] = f"Exception: {str(e)}"
                debug_info.append(user_info)
                self.fallback_counters["exception_occurred"] += 1
                random_candidates = np.random.choice(
                    self.post_embeddings_df['post_id'].unique(), 
                    size=self.n_candidates, 
                    replace=False
                )
                candidates_dict[batch_node_interact_times[i]] = random_candidates
        
        # Save debug info for analysis
        self.debug_info = debug_info
        
        # # Print debug statistics
        # if self.total_processed % 100 == 0:
        #     print(f"Debug stats: Total processed: {self.total_processed}")
        #     print(f"True post added count: {self.true_post_added_count} ({self.true_post_added_count/self.total_processed*100:.2f}%)")
            
        #     # Print hit rate statistics
        #     print("Hit Rate@k:")
        #     for k in self.k_values:
        #         hit_rate = (self.hit_counters[k] / self.total_processed) * 100
        #         print(f"  Hit@{k}: {hit_rate:.2f}%")
            
        #     # Print detailed fallback statistics
        #     total_fallbacks = sum(self.fallback_counters.values())
        #     print(f"Total fallbacks: {total_fallbacks} ({total_fallbacks/self.total_processed*100:.2f}%)")
        #     print("Fallback reasons breakdown:")
        #     for reason, count in self.fallback_counters.items():
        #         if count > 0:
        #             print(f"  - {reason}: {count} ({count/total_fallbacks*100:.2f}% of fallbacks)")
            
        #     # Analyze why true posts aren't in candidates
        #     if len(debug_info) > 0:
        #         not_in_candidates = [info for info in debug_info if info.get("true_post_in_candidates") is False]
        #         if not_in_candidates:
        #             print(f"Sample reasons true post not in candidates:")
        #             for i, info in enumerate(not_in_candidates[:3]):
        #                 print(f"  Example {i+1}: {info.get('error', 'No error')}, Active posts: {info.get('num_active_posts', 'N/A')}")
        
        return candidates_dict