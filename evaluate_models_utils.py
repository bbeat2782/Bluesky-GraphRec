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

from utils.metrics import get_link_prediction_metrics
from utils.utils import set_random_seed
from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data


def evaluate_real(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler, evaluate_data: Data,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    #evaluate_neg_edge_sampler.reset_random_state()

    model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()
    with torch.no_grad():
        # store evaluate losses and metrics
        mrr_results = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            candidates_dict = evaluate_neg_edge_sampler.sample(
                len(batch_src_node_ids), 
                batch_src_node_ids, 
                batch_dst_node_ids, 
                batch_node_interact_times
            )

            # Prepare for batch processing
            batch_candidates = []
            batch_interact_times = []
            batch_src_ids = []

            for src_id, interact_time in zip(batch_src_node_ids, batch_node_interact_times):
                candidate_ids = candidates_dict[interact_time]
                batch_candidates.append(list(candidate_ids))
                batch_interact_times.append([interact_time] * len(candidate_ids))
                batch_src_ids.append([src_id] * len(candidate_ids))

            # Flatten batch data for processing
            batch_candidates = np.concatenate(batch_candidates)
            batch_interact_times = np.concatenate(batch_interact_times)
            batch_src_ids = np.concatenate(batch_src_ids)

            # Compute embeddings in one operation
            src_embeddings, dst_embeddings, _ = model[0].compute_src_dst_node_temporal_embeddings(
                src_node_ids=batch_src_ids,
                dst_node_ids=batch_candidates,
                node_interact_times=batch_interact_times,
                is_eval=True
            )

            # Compute scores for all user-candidate pairs in the batch
            probabilities = model[1](input_1=src_embeddings, input_2=dst_embeddings).squeeze(dim=-1).sigmoid()

            # Reshape probabilities to group by users
            split_indices = np.cumsum([len(candidates_dict[interact_time]) for interact_time in batch_node_interact_times])
            #split_indices = np.cumsum([len(candidate_ids) for candidate_ids in list(candidates_dict.values())[:-1]])
            grouped_probabilities = np.split(probabilities.cpu().numpy(), split_indices)
            #print('grouped_probabilities', grouped_probabilities.shape)
            grouped_candidates = np.split(batch_candidates, split_indices)
            # grouped_history = [
            #     src_nodes_neighbor_ids_list[start:end] 
            #     for start, end in zip([0] + list(split_indices), split_indices + [len(src_nodes_neighbor_ids_list)])
            # ]
            # print('grouped_history', len(grouped_history))
            # raise ValueError()

            #print('grouped_candidates', grouped_candidates.shape)

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

    avg_mrr = sum(mrr_results) / len(mrr_results)
    print(f"Mean Reciprocal Rank (MRR): {avg_mrr}")

    return avg_mrr


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    if model_name in ['GraphRec']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            # print('evaluate_data_indices', evaluate_data_indices)
            # raise ValueError()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            if evaluate_neg_edge_sampler.negative_sample_strategy != 'random':
                batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids),
                                                                                                  batch_src_node_ids=batch_src_node_ids,
                                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                                  current_batch_start_time=batch_node_interact_times[0],
                                                                                                  current_batch_end_time=batch_node_interact_times[-1])
            else:
                _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['GraphRec']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())

            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics
