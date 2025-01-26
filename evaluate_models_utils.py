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

    if model_name in ['GraphRec']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        mrr_results = []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            # print('evaluate_data_indices', evaluate_data_indices)
            # raise ValueError()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            # print('batch_src_node_ids', batch_src_node_ids.shape)
            # print('batch_dst_node_ids', batch_dst_node_ids.shape)
            # print('batch_node_interact_times', batch_node_interact_times.shape)
            # print('batch_edge_ids', batch_edge_ids.shape)
            
            candidates_dict = evaluate_neg_edge_sampler.sample(len(batch_src_node_ids), batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times)
            
            # print(candidates_dict.keys())

            for src_id, interact_time, true_dst_id in zip(batch_src_node_ids, batch_node_interact_times, batch_dst_node_ids):
                # print('src_id:', src_id)
                # print('interact_time:', interact_time)
                # print('true_dst_id:', true_dst_id)
                candidate_ids = candidates_dict[interact_time]
                # print('candidate length:', len(candidate_ids))

                # Prepare batch data for this user and time
                batch_src_node_ids = np.array(list(candidate_ids))  # Candidate IDs for this interact time
                batch_node_interact_times = np.array([interact_time] * len(candidate_ids))  # Same time for all candidates


                src_for_candidates = np.array([src_id] * len(candidate_ids))  # Repeat src_id to match candidate shape
                candidates = np.array(list(candidate_ids))  # Candidate IDs as a 1D array
                interact_time_for_candidates = np.array([interact_time] * len(candidate_ids))  # Same time for all candidates

                # Ensure shapes are consistent
                assert src_for_candidates.shape == (len(candidate_ids),)
                assert candidates.shape == (len(candidate_ids),)
                assert interact_time_for_candidates.shape == (len(candidate_ids),)
                batch_size = 400
                probabilities_list = []
                candidate_list = []
                #for i in tqdm(range(0, len(candidates), batch_size), desc=f"Processing batches for src_id {src_id}"):
                for i in range(0, len(candidates), batch_size):
                    # Slice data for the current batch
                    batch_src = src_for_candidates[i:min(i + batch_size, len(candidates))]
                    batch_dst = candidates[i:min(i + batch_size, len(candidates))]
                    batch_times = interact_time_for_candidates[i:min(i + batch_size, len(candidates))]

                    # Compute embeddings for the current batch
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src,
                        dst_node_ids=batch_dst,
                        node_interact_times=batch_times
                    )

                    probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                    probabilities_list.append(probabilities.cpu().numpy())
                    candidate_list.append(batch_dst)

                # Flatten probabilities and candidate lists
                probabilities_array = np.concatenate(probabilities_list)
                candidate_array = np.concatenate(candidate_list)

                # Sort candidates by probabilities in descending order
                sorted_indices = np.argsort(-probabilities_array)  # Negative sign for descending order
                sorted_candidates = candidate_array[sorted_indices]
                sorted_probabilities = probabilities_array[sorted_indices]


                # Calculate the rank of the true destination
                if true_dst_id in sorted_candidates:
                    rank = np.where(sorted_candidates == true_dst_id)[0][0] + 1  # 1-based rank
                    mrr_results.append(1 / rank)
                else:
                    mrr_results.append(0)  # True destination not found
                            
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
