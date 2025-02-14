import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import time
import argparse
import os
import json
import matplotlib.pyplot as plt

from utils.metrics import get_link_prediction_metrics
from utils.utils import set_random_seed
from utils.utils import NeighborSampler
from utils.DataLoader import Data


def evaluate_real(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler, evaluate_data: Data,
                                   num_neighbors: int = 20):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: CandidateEdgeSampler, evaluate candidate edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param num_neighbors: int, number of neighbors to sample for each node
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)

    model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()
    candidates_length = {}
    recommended_posts = []

    with torch.no_grad():
        # store evaluate losses and metrics
        mrr_results = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            # For dynamic features
            batch_src_idx = evaluate_data.idx[evaluate_data_indices]
            
            popularity_based = False  # TODO make this as an argument
            if popularity_based:
                model_name = 'Popularity'
                candidates_dict = evaluate_neg_edge_sampler.sample(
                    len(batch_src_node_ids), 
                    batch_src_node_ids, 
                    batch_dst_node_ids, 
                    batch_node_interact_times,
                    popularity_based=popularity_based
                )

                for true_dst_id, interact_time in zip(batch_dst_node_ids, batch_node_interact_times):
                    candidates = candidates_dict[interact_time]

                    # Find the rank of the true destination ID
                    if true_dst_id in candidates:
                        rank = np.where(candidates == true_dst_id)[0][0] + 1
                        reciprocal_rank = 1.0 / rank
                    else:
                        reciprocal_rank = 0.0  # True ID not in candidates
        
                    mrr_results.append(reciprocal_rank)
                    recommended_posts.append(candidates.tolist())
            else:
                candidates_dict = evaluate_neg_edge_sampler.sample(
                    len(batch_src_node_ids), 
                    batch_src_node_ids, 
                    batch_dst_node_ids, 
                    batch_node_interact_times
                )
    
                # Iterate through candidates_dict to calculate lengths
                for start_time, candidates in candidates_dict.items():
                    # Store in candidates_length (accumulate counts if start_time repeats across batches)
                    start_time = str(start_time)
                    if start_time not in candidates_length:
                        # Get the length of candidates for the given start_time
                        num_candidates = len(candidates)
                        candidates_length[start_time] = num_candidates
    
                # Prepare for batch processing
                batch_candidates = []
                batch_interact_times = []
                batch_src_ids = []
                batch_idx = []
    
                for src_id, interact_time, src_idx in zip(batch_src_node_ids, batch_node_interact_times, batch_src_idx):
                    candidate_ids = candidates_dict[interact_time]
                    batch_candidates.append(list(candidate_ids))
                    batch_interact_times.append([interact_time] * len(candidate_ids))
                    batch_src_ids.append([src_id] * len(candidate_ids))
                    batch_idx.append([src_idx] * len(candidate_ids))
    
                # Flatten batch data for processing
                batch_candidates = np.concatenate(batch_candidates)
                batch_interact_times = np.concatenate(batch_interact_times)
                batch_src_ids = np.concatenate(batch_src_ids)
                batch_idx = np.concatenate(batch_idx)

                if model_name in {'GraphRec', 'GraphRecMulti', 'GraphRecMultiCo'}:
                    # Compute embeddings in one operation
                    src_embeddings, dst_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_ids,
                        dst_node_ids=batch_candidates,
                        node_interact_times=batch_interact_times,
                        batch_src_idx=batch_idx
                    )
                elif model_name == 'TGAT':
                    # Compute embeddings in one operation
                    src_embeddings, dst_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_ids,
                        dst_node_ids=batch_candidates,
                        node_interact_times=batch_interact_times,
                        num_neighbors=num_neighbors
                    )
    
                # Compute scores for all user-candidate pairs in the batch
                probabilities = model[1](input_1=src_embeddings, input_2=dst_embeddings).squeeze(dim=-1).sigmoid()
    
                # Reshape probabilities to group by users
                split_indices = np.cumsum([len(candidates_dict[interact_time]) for interact_time in batch_node_interact_times])
                grouped_probabilities = np.split(probabilities.cpu().numpy(), split_indices)
                grouped_candidates = np.split(batch_candidates, split_indices)
    
                # Evaluate MRR for each user in the batch
                for post_probabilities, post_candidates, true_dst_id in zip(grouped_probabilities, grouped_candidates, batch_dst_node_ids):
                    # Convert to numpy for indexing
                    post_probabilities = np.array(post_probabilities)
                    post_candidates = np.array(post_candidates)
                    
                    # Find the index of the true destination ID
                    true_dst_index = np.where(post_candidates == true_dst_id)[0]
                    
                    if len(true_dst_index) > 0:  # Ensure the true destination exists
                        true_dst_index = true_dst_index[0]
                        true_dst_probability = post_probabilities[true_dst_index]
                        
                        # Count how many probabilities are higher than the true_dst_probability
                        rank = 1 + np.sum(post_probabilities > true_dst_probability)
                        mrr_results.append(1 / rank)
                    else:
                        # True destination not found in candidates
                        mrr_results.append(0)

                    # NOTE: For checking which posts are recommended. Comment this when you do not need it
                    # Sort indices based on probabilities in descending order
                    sorted_indices = np.argsort(-post_probabilities)  # Negative sign for descending order
                
                    # Apply sorted indices to both arrays
                    sorted_candidates = post_candidates[sorted_indices]

                    recommended_posts.append(sorted_candidates.tolist()) 

    # NOTE: For checking which posts are recommended. Comment this when you do not need it
    # Create directory if it doesn't exist
    save_dir = f"saved_results/{model_name}/bluesky"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to JSON file
    save_path = os.path.join(save_dir, "recommended_posts.json")
    with open(save_path, "w") as json_file:
        json.dump(recommended_posts, json_file, indent=4)

    np.save(f"saved_results/{model_name}/bluesky/mrr_results.npy", np.array(mrr_results))
    avg_mrr = sum(mrr_results) / len(mrr_results)
    
    print(f"Mean Reciprocal Rank (MRR): {avg_mrr}")

    # Save the candidates_length dictionary to a JSON file
    output_dict_path = f"saved_results/{model_name}/bluesky/candidates_length.json"
    with open(output_dict_path, 'w') as f:
       json.dump(candidates_length, f, indent=4)

    # Convert the dictionary values into a flat list of counts
    all_counts = list(candidates_length.values())
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(all_counts, bins='auto', edgecolor='black')
    plt.title("Distribution of Candidate Counts")
    plt.xlabel("Number of Candidates")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save the plot
    output_path = f"saved_results/{model_name}/bluesky/candidates_length_histogram.png"
    plt.savefig(output_path)
    plt.close()

    return avg_mrr


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler, evaluate_data: Data,
                                   num_neighbors: int = 20):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: MultipleNegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param num_neighbors: int, number of neighbors to sample for each node
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['GraphRec', 'TGAT', 'GraphRecMulti', 'GraphRecMultiCo']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    subset_fraction = 0.1  # Eval on 30% of the data
    num_batches = len(evaluate_idx_data_loader)
    start_batch = int(num_batches * (1 - subset_fraction))  # Compute start index for last 30%

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            if batch_idx < start_batch:
                continue  # Skip first 70% of batches
            
            evaluate_data_indices = evaluate_data_indices.numpy()
            # print('evaluate_data_indices', evaluate_data_indices)
            # raise ValueError()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            # For dynamic features
            batch_src_idx = evaluate_data.idx[evaluate_data_indices]

            _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids), current_batch_start_time=batch_node_interact_times)
            # batch_neg_src_node_ids = batch_src_node_ids
            batch_neg_src_node_ids = np.repeat(batch_src_node_ids, 4, axis=0).reshape(len(batch_src_node_ids), 4)
            batch_neg_src_idx = np.repeat(batch_src_idx, 4, axis=0).reshape(len(batch_src_idx), 4)

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['GraphRec', 'GraphRecMulti', 'GraphRecMultiCo']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      batch_src_idx=batch_src_idx)

            
                # Flatten negative samples to compute embeddings properly
                batch_neg_src_node_ids_flat = batch_neg_src_node_ids.flatten()  # (batch_size * 4,)
                batch_neg_dst_node_ids_flat = batch_neg_dst_node_ids.flatten()  # (batch_size * 4,)
                batch_neg_times_flat = np.repeat(batch_node_interact_times, 4, axis=0).flatten()  # (batch_size * 4,)
                batch_neg_src_idx_flat = batch_neg_src_idx.flatten()  # (batch_size * 4,)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids_flat,
                                                                      dst_node_ids=batch_neg_dst_node_ids_flat,
                                                                      node_interact_times=batch_neg_times_flat,
                                                                      batch_src_idx=batch_neg_src_idx_flat)

                # Reshape back to (batch_size, 4, node_feat_dim) so that each positive has 4 negatives
                node_feat_dim = batch_neg_src_node_embeddings.shape[1]  # Get feature dimension
                batch_neg_src_node_embeddings = batch_neg_src_node_embeddings.reshape(len(batch_src_node_ids), 4, node_feat_dim)
                batch_neg_dst_node_embeddings = batch_neg_dst_node_embeddings.reshape(len(batch_src_node_ids), 4, node_feat_dim)
            elif model_name in ['TGAT', 'CAWN']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # Flatten negative samples to compute embeddings properly
                batch_neg_src_node_ids_flat = batch_neg_src_node_ids.flatten()  # (batch_size * 4,)
                batch_neg_dst_node_ids_flat = batch_neg_dst_node_ids.flatten()  # (batch_size * 4,)
                batch_neg_times_flat = np.repeat(batch_node_interact_times, 4, axis=0).flatten()  # (batch_size * 4,)
                batch_neg_src_idx_flat = batch_neg_src_idx.flatten()  # (batch_size * 4,)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids_flat,
                                                                      dst_node_ids=batch_neg_dst_node_ids_flat,
                                                                      node_interact_times=batch_neg_times_flat,
                                                                      num_neighbors=num_neighbors)

                # Reshape back to (batch_size, 4, node_feat_dim) so that each positive has 4 negatives
                node_feat_dim = batch_neg_src_node_embeddings.shape[1]  # Get feature dimension
                batch_neg_src_node_embeddings = batch_neg_src_node_embeddings.reshape(len(batch_src_node_ids), 4, node_feat_dim)
                batch_neg_dst_node_embeddings = batch_neg_dst_node_embeddings.reshape(len(batch_src_node_ids), 4, node_feat_dim)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")

            embedding_dim = batch_src_node_embeddings.shape[1]
            # Flatten negatives for processing
            batch_neg_src_node_embeddings_flat = batch_neg_src_node_embeddings.view(-1, embedding_dim)  # (batch_size * 4, embedding_dim)
            batch_neg_dst_node_embeddings_flat = batch_neg_dst_node_embeddings.view(-1, embedding_dim)  # (batch_size * 4, embedding_dim)

            positive_scores = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1)
            negative_scores = model[1](
                input_1=batch_neg_src_node_embeddings_flat, 
                input_2=batch_neg_dst_node_embeddings_flat
            ).squeeze(dim=-1).view(positive_scores.shape[0], 4)  # Reshape back to (batch_size, 4)

            # Apply BPR loss: Maximize positive score over all negatives
            bpr_loss = -torch.log(torch.sigmoid(positive_scores.unsqueeze(1) - negative_scores) + 1e-8).mean()

            predicts = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)  # (batch_size, 5)
            labels = torch.cat([torch.ones(positive_scores.shape[0], 1), torch.zeros(positive_scores.shape[0], 4)], dim=1)

            evaluate_losses.append(bpr_loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts, labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {bpr_loss.item()}')

    return evaluate_losses, evaluate_metrics
