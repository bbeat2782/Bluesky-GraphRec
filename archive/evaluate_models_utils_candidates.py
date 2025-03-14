# Standard library imports
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
from torch.utils.data import DataLoader  # For batching data
from tqdm import tqdm  # Progress bar
import numpy as np  # Numerical computations
import logging  # Logging utilities
import time  # Time utilities
import argparse  # Command line argument parsing
import os  # Operating system utilities
import json  # JSON file handling
import matplotlib.pyplot as plt  # Plotting

# Custom utility imports
from utils.metrics import get_link_prediction_metrics  # Metrics for link prediction
from utils.utils import set_random_seed, NeighborSampler  # Random seed and neighbor sampling
from utils.DataLoader import Data  # Custom data loading
from utils.candidates import dummy_candidate_generator  # Generates candidate nodes


def evaluate_real(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                 evaluate_neg_edge_sampler, evaluate_data: Data,
                 num_neighbors: int = 20, time_gap=8):
    """
    Evaluates models on the link prediction task using real-world data.
    
    This function takes a trained model and evaluates its performance on predicting links/interactions
    between nodes in a temporal graph. It uses Mean Reciprocal Rank (MRR) as the evaluation metric.
    
    Example data shapes:
    - batch_src_node_ids: [batch_size] e.g. [1, 4, 7] (user IDs)
    - batch_dst_node_ids: [batch_size] e.g. [2, 5, 8] (post IDs they interacted with)
    - batch_node_interact_times: [batch_size] e.g. [100, 101, 102] (timestamps)
    
    ASCII visualization of the evaluation process:
    
    User 1 -----> Post A (true interaction)
         \
          \---> [Post B, Post C, Post D] (candidate posts)
    
    For each user, we:
    1. Get their true interaction (Post A)
    2. Generate candidate posts they could have interacted with
    3. Rank all posts (true + candidates) by model prediction
    4. Calculate reciprocal rank based on true post's position
    
    Parameters:
    -----------
    model_name : str
        Name of the model being evaluated ('GraphRec', 'TGAT', etc.)
    model : nn.Module
        The PyTorch model to evaluate
    neighbor_sampler : NeighborSampler
        Object that samples neighboring nodes for message passing
    evaluate_idx_data_loader : DataLoader
        Batched data loader containing evaluation indices
    evaluate_neg_edge_sampler : object
        Object that generates negative/candidate edges for evaluation
    evaluate_data : Data
        Contains all graph data including nodes, edges, timestamps
    num_neighbors : int, default=20
        Number of neighbors to sample for each node
    time_gap : int, default=8
        Time window for considering interactions
    """

    # Set the neighbor sampler for the model's first component (usually the encoder)
    model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()  # Put model in evaluation mode
    
    # Dictionary to store number of candidates per timestamp
    candidates_length = {}
    
    # List to store recommended posts for each user
    # Shape: [num_users, num_candidates] e.g. [[1,2,3], [4,5,6]]
    recommended_posts = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        mrr_results = []  # Store Mean Reciprocal Rank results
        
        # Progress bar for evaluation batches
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        
        # Process each batch of evaluation indices
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            # Convert indices to numpy for indexing
            evaluate_data_indices = evaluate_data_indices.numpy()
            
            # Extract batch data using indices
            # Each has shape [batch_size]
            batch_src_node_ids = evaluate_data.src_node_ids[evaluate_data_indices]  # Source/user nodes
            batch_dst_node_ids = evaluate_data.dst_node_ids[evaluate_data_indices]  # Destination/post nodes
            batch_node_interact_times = evaluate_data.node_interact_times[evaluate_data_indices]  # Interaction timestamps
            batch_edge_ids = evaluate_data.edge_ids[evaluate_data_indices]  # Edge IDs
            batch_src_idx = evaluate_data.idx[evaluate_data_indices]  # Source indices for dynamic features

            # Flag for using popularity-based recommendation (baseline)
            popularity_based = False  # TODO: make this an argument
            
            if popularity_based:
                model_name = 'Popularity'
                
                # Generate candidate posts for each timestamp
                # Returns dict: {timestamp: [candidate_ids]}
                candidates_dict = dummy_candidate_generator(batch_src_node_ids, batch_node_interact_times)

                # Debug logging of candidates
                with open('candidates_dict_candidates_debug.txt', 'w') as f:
                    total_candidates = sum(len(candidates) for candidates in candidates_dict.values())
                    f.write(f'Total number of candidates across all times: {total_candidates}\n')
                    f.write(f'Number of unique times: {len(candidates_dict)}\n\n')
                    
                    f.write('Candidates shapes by time:\n')
                    for time, candidates in candidates_dict.items():
                        f.write(f'  time {time}: candidates shape {np.array(list(candidates)).shape}\n')
                    
                    f.write('\nFull candidates dictionary:\n')
                    f.write(str(candidates_dict))

                # Print shapes for debugging
                print('candidates_dict shapes:')
                for time, candidates in candidates_dict.items():
                    print(f'  time {time}: candidates shape {np.array(list(candidates)).shape}')
                print('candidates_dict:', candidates_dict)

                # Write full dictionary for debugging
                with open('candidates_dict_debug.txt', 'w') as f:
                    f.write(str(candidates_dict))

                print('candidates_dict', candidates_dict)

                # Calculate MRR for each true interaction
                for true_dst_id, interact_time in zip(batch_dst_node_ids, batch_node_interact_times):
                    candidates = candidates_dict[interact_time]

                    # Find rank of true destination in candidates
                    if true_dst_id in candidates:
                        rank = np.where(candidates == true_dst_id)[0][0] + 1
                        reciprocal_rank = 1.0 / rank
                    else:
                        reciprocal_rank = 0.0  # True ID not found
        
                    mrr_results.append(reciprocal_rank)
                    recommended_posts.append(candidates.tolist())
                    
            else:  # Model-based recommendation

                with open('batch_data_debug.txt', 'w') as f:
                    f.write('batch_src_node_ids:\n')
                    f.write(str(batch_src_node_ids) + '\n\n')
                    f.write('batch_dst_node_ids:\n') 
                    f.write(str(batch_dst_node_ids) + '\n\n')
                    f.write('batch_node_interact_times:\n')
                    f.write(str(batch_node_interact_times) + '\n\n')

                # Generate candidates similar to popularity-based approach
                candidates_dict = dummy_candidate_generator(batch_src_node_ids, batch_node_interact_times)

                # Debug logging (same as above)
                with open('candidates_dict_candidates_debug.txt', 'w') as f:
                    total_candidates = sum(len(candidates) for candidates in candidates_dict.values())
                    f.write(f'Total number of candidates across all times: {total_candidates}\n')
                    f.write(f'Number of unique times: {len(candidates_dict)}\n\n')
                    
                    f.write('Candidates shapes by time:\n')
                    for time, candidates in candidates_dict.items():
                        f.write(f'  time {time}: candidates shape {np.array(list(candidates)).shape}\n')
                    
                    f.write('\nFull candidates dictionary:\n')
                    f.write(str(candidates_dict))

                print('candidates_dict shapes:')
                for time, candidates in candidates_dict.items():
                    print(f'  time {time}: candidates shape {np.array(list(candidates)).shape}')
                print('candidates_dict:', candidates_dict)
    
                # Store number of candidates per timestamp
                for start_time, candidates in candidates_dict.items():
                    start_time = str(start_time)
                    if start_time not in candidates_length:
                        candidates_length[start_time] = len(candidates)
    
                # Prepare batch data for model processing
                # Lists to store expanded batch data where each user-candidate pair is a row
                batch_candidates = []  # All candidate posts
                batch_interact_times = []  # Repeated timestamps for each candidate
                batch_src_ids = []  # Repeated user IDs for each candidate
                batch_src_ids_no_duplicates = []  # Original user IDs
                batch_idx = []  # Repeated indices for each candidate
    
                # Create expanded arrays for batch processing
                for src_id, interact_time, src_idx in zip(batch_src_node_ids, batch_node_interact_times, batch_src_idx):
                    candidate_ids = candidates_dict[interact_time]
                    batch_candidates.append(list(candidate_ids))
                    batch_interact_times.append([interact_time] * len(candidate_ids))
                    batch_src_ids.append([src_id] * len(candidate_ids))
                    batch_src_ids_no_duplicates.append(src_id)
                    batch_idx.append([src_idx] * len(candidate_ids))
    
                # Flatten all batch arrays
                # Shape: [total_candidates_across_batch]
                batch_candidates = np.concatenate(batch_candidates)
                batch_interact_times = np.concatenate(batch_interact_times)
                batch_src_ids = np.concatenate(batch_src_ids)
                batch_idx = np.concatenate(batch_idx)

                # Compute embeddings based on model type
                if model_name in {'GraphRec', 'GraphRecMulti', 'GraphRecMultiCo'}:
                    # Get embeddings for all user-candidate pairs
                    # Shape: [total_candidates_across_batch, embedding_dim]
                    src_embeddings, dst_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_ids,
                        dst_node_ids=batch_candidates,
                        node_interact_times=batch_interact_times,
                        batch_src_idx=batch_idx
                    )
                elif model_name == 'TGAT':
                    src_embeddings, dst_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_ids,
                        dst_node_ids=batch_candidates,
                        node_interact_times=batch_interact_times,
                        num_neighbors=num_neighbors
                    )
                else:
                    raise ValueError(f"Wrong value for model_name {model_name}!")
    
                # Compute interaction probabilities for all pairs
                # Shape: [total_candidates_across_batch]
                probabilities = model[1](input_1=src_embeddings, input_2=dst_embeddings).squeeze(dim=-1).sigmoid()
    
                # Split probabilities and candidates back into per-user groups
                split_indices = np.cumsum([len(candidates_dict[interact_time]) for interact_time in batch_node_interact_times])
                grouped_probabilities = np.split(probabilities.cpu().numpy(), split_indices)
                grouped_candidates = np.split(batch_candidates, split_indices)
    
                # Calculate MRR for each user
                for post_probabilities, post_candidates, true_dst_id, src_id in zip(
                    grouped_probabilities, grouped_candidates, batch_dst_node_ids, batch_src_ids_no_duplicates):
                    
                    post_probabilities = np.array(post_probabilities)
                    post_candidates = np.array(post_candidates)
                    
                    # Find true post's index in candidates
                    true_dst_index = np.where(post_candidates == true_dst_id)[0]
                    
                    if len(true_dst_index) > 0:
                        true_dst_index = true_dst_index[0]
                        true_dst_probability = post_probabilities[true_dst_index]
                        
                        # Rank = 1 + number of candidates with higher probability
                        rank = 1 + np.sum(post_probabilities > true_dst_probability)
                        mrr_results.append(1 / rank)
                    else:
                        mrr_results.append(0)  # True post not in candidates

                    # Store recommended posts sorted by probability
                    sorted_indices = np.argsort(-post_probabilities)
                    sorted_candidates = post_candidates[sorted_indices]
                    recommended_posts.append(sorted_candidates.tolist())

    # Save results and create visualizations
    save_dir = f"saved_results/{model_name}/bluesky"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save recommended posts
    save_path = os.path.join(save_dir, "recommended_posts.json")
    with open(save_path, "w") as json_file:
        json.dump(recommended_posts, json_file, indent=4)

    # Save MRR results
    np.save(f"saved_results/{model_name}/bluesky/mrr_results.npy", np.array(mrr_results))
    avg_mrr = sum(mrr_results) / len(mrr_results)
    
    print(f"Mean Reciprocal Rank (MRR): {avg_mrr}")

    # Save candidates per timestamp statistics
    output_dict_path = f"saved_results/{model_name}/bluesky/candidates_length.json"
    with open(output_dict_path, 'w') as f:
       json.dump(candidates_length, f, indent=4)

    # Create histogram of candidate counts
    all_counts = list(candidates_length.values())
    plt.figure(figsize=(10, 6))
    plt.hist(all_counts, bins='auto', edgecolor='black')
    plt.title("Distribution of Candidate Counts")
    plt.xlabel("Number of Candidates")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save histogram
    output_path = f"saved_results/{model_name}/bluesky/candidates_length_histogram.png"
    plt.savefig(output_path)
    plt.close()

    return avg_mrr


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler, evaluate_data: Data,
                                   num_neighbors: int = 20, time_gap=2000):
    """
    Evaluates models on the link prediction task using negative sampling.
    
    This function evaluates how well the model can distinguish between real interactions (positive edges)
    and randomly sampled fake interactions (negative edges). It uses BPR loss for training.
    
    ASCII visualization of negative sampling:
    
    Positive:
    User 1 -----> Post A (true interaction)
    
    Negatives:
    User 1 -----> Post B (fake interaction)
    User 1 -----> Post C (fake interaction)
    User 1 -----> Post D (fake interaction)
    User 1 -----> Post E (fake interaction)
    
    For each positive edge, we sample 4 negative edges by keeping the same user
    but replacing the post with a random one.
    
    Data shapes example:
    - batch_src_node_ids: [32] (batch of 32 users)
    - batch_dst_node_ids: [32] (batch of 32 posts they interacted with)
    - batch_neg_dst_node_ids: [32, 4] (4 negative posts per user)
    
    Parameters:
    -----------
    [same as evaluate_real()]
    """
    
    # Set fixed seed for reproducible negative sampling
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    # Set neighbor sampler for message passing models
    if model_name in ['GraphRec', 'TGAT', 'GraphRecMulti', 'GraphRecMultiCo']:
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    # Evaluate on subset of data for efficiency
    subset_fraction = 0.1  # Use 10% of data
    num_batches = len(evaluate_idx_data_loader)
    start_batch = int(num_batches * (1 - subset_fraction))

    with torch.no_grad():
        evaluate_losses = []
        evaluate_metrics = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            if batch_idx < start_batch:
                continue  # Skip first 90% of batches
            
            # Get batch data
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids = evaluate_data.src_node_ids[evaluate_data_indices]
            batch_dst_node_ids = evaluate_data.dst_node_ids[evaluate_data_indices]
            batch_node_interact_times = evaluate_data.node_interact_times[evaluate_data_indices]
            batch_edge_ids = evaluate_data.edge_ids[evaluate_data_indices]
            batch_src_idx = evaluate_data.idx[evaluate_data_indices]

            # Sample negative edges
            # Returns [batch_size, 4] negative posts for each user
            _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(
                size=len(batch_src_node_ids), 
                current_batch_start_time=batch_node_interact_times
            )
            
            # Repeat source nodes for negative samples
            # [batch_size] -> [batch_size, 4]
            batch_neg_src_node_ids = np.repeat(batch_src_node_ids, 4, axis=0).reshape(len(batch_src_node_ids), 4)
            batch_neg_src_idx = np.repeat(batch_src_idx, 4, axis=0).reshape(len(batch_src_idx), 4)

            # Compute embeddings based on model type
            if model_name in ['GraphRec', 'GraphRecMulti', 'GraphRecMultiCo']:
                # Get embeddings for positive pairs
                # Shape: [batch_size, embedding_dim]
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        batch_src_idx=batch_src_idx
                    )

                # Flatten negative samples for embedding computation
                batch_neg_src_node_ids_flat = batch_neg_src_node_ids.flatten()  # [batch_size * 4]
                batch_neg_dst_node_ids_flat = batch_neg_dst_node_ids.flatten()  # [batch_size * 4]
                batch_neg_times_flat = np.repeat(batch_node_interact_times, 4, axis=0).flatten()  # [batch_size * 4]
                batch_neg_src_idx_flat = batch_neg_src_idx.flatten()  # [batch_size * 4]

                # Get embeddings for negative pairs
                # Shape: [batch_size * 4, embedding_dim]
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids_flat,
                        dst_node_ids=batch_neg_dst_node_ids_flat,
                        node_interact_times=batch_neg_times_flat,
                        batch_src_idx=batch_neg_src_idx_flat
                    )

                # Reshape negative embeddings to group by positive example
                # Shape: [batch_size, 4, embedding_dim]
                node_feat_dim = batch_neg_src_node_embeddings.shape[1]
                batch_neg_src_node_embeddings = batch_neg_src_node_embeddings.reshape(
                    len(batch_src_node_ids), 4, node_feat_dim)
                batch_neg_dst_node_embeddings = batch_neg_dst_node_embeddings.reshape(
                    len(batch_src_node_ids), 4, node_feat_dim)
                
            elif model_name in ['TGAT']:
                # Similar process for TGAT model
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=num_neighbors
                    )

                batch_neg_src_node_ids_flat = batch_neg_src_node_ids.flatten()
                batch_neg_dst_node_ids_flat = batch_neg_dst_node_ids.flatten()
                batch_neg_times_flat = np.repeat(batch_node_interact_times, 4, axis=0).flatten()
                batch_neg_src_idx_flat = batch_neg_src_idx.flatten()

                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids_flat,
                        dst_node_ids=batch_neg_dst_node_ids_flat,
                        node_interact_times=batch_neg_times_flat,
                        num_neighbors=num_neighbors
                    )

                node_feat_dim = batch_neg_src_node_embeddings.shape[1]
                batch_neg_src_node_embeddings = batch_neg_src_node_embeddings.reshape(
                    len(batch_src_node_ids), 4, node_feat_dim)
                batch_neg_dst_node_embeddings = batch_neg_dst_node_embeddings.reshape(
                    len(batch_src_node_ids), 4, node_feat_dim)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            # Flatten negative embeddings for scoring
            embedding_dim = batch_src_node_embeddings.shape[1]
            batch_neg_src_node_embeddings_flat = batch_neg_src_node_embeddings.view(-1, embedding_dim)
            batch_neg_dst_node_embeddings_flat = batch_neg_dst_node_embeddings.view(-1, embedding_dim)

            # Compute interaction scores
            # Shape: [batch_size]
            positive_scores = model[1](
                input_1=batch_src_node_embeddings, 
                input_2=batch_dst_node_embeddings
            ).squeeze(dim=-1)
            
            # Shape: [batch_size, 4]
            negative_scores = model[1](
                input_1=batch_neg_src_node_embeddings_flat, 
                input_2=batch_neg_dst_node_embeddings_flat
            ).squeeze(dim=-1).view(positive_scores.shape[0], 4)

            # Compute BPR loss
            # For each positive, maximize its score over all its negatives
            bpr_loss = -torch.log(
                torch.sigmoid(positive_scores.unsqueeze(1) - negative_scores) + 1e-8
            ).mean()

            # Combine positive and negative scores
            # Shape: [batch_size, 5] (1 positive + 4 negatives per row)
            predicts = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
            
            # Create labels (1 for positive, 0 for negatives)
            # Shape: [batch_size, 5]
            labels = torch.cat([
                torch.ones(positive_scores.shape[0], 1), 
                torch.zeros(positive_scores.shape[0], 4)
            ], dim=1)

            evaluate_losses.append(bpr_loss.item())
            evaluate_metrics.append(get_link_prediction_metrics(predicts, labels))

            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {bpr_loss.item()}'
            )

    return evaluate_losses, evaluate_metrics
