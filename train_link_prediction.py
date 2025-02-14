import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from torch.amp import autocast

from models.TGAT import TGAT
from models.GraphRec import GraphRec
from models.GraphRecMulti import GraphRecMulti
from models.GraphRecMultiCo import GraphRecMultiCo
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer, save_plot
from utils.utils import get_neighbor_sampler, MultipleNegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # Suppress Matplotlib debug messages once at the beginning
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, user_dynamic_features = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)
    
    # # Write neighbor sampler data to file
    # with open('neighbor_sampler_data.txt', 'w') as f:
    #     f.write("\n\nFirst 5 entries of neighbor IDs:\n") 
    #     f.write(str(train_neighbor_sampler.nodes_neighbor_ids[:5]))
    #     f.write("\n\nFirst 5 entries of edge IDs:\n")
    #     f.write(str(train_neighbor_sampler.nodes_edge_ids[:5]))
    #     f.write("\n\nFirst 5 entries of neighbor timestamps:\n")
    #     f.write(str(train_neighbor_sampler.nodes_neighbor_times[:5]))
    #     f.write("\n\nFirst 5 entries of neighbor indices:\n")
    #     f.write(str(train_neighbor_sampler.nodes_neighbor_idx[:5]))

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    train_neg_edge_sampler = MultipleNegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids, seed=2025, negative_sample_strategy=args.negative_sample_strategy, interact_times=train_data.node_interact_times)
    val_neg_edge_sampler = MultipleNegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0, negative_sample_strategy=args.negative_sample_strategy, interact_times=full_data.node_interact_times)
    new_node_val_neg_edge_sampler = MultipleNegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1, negative_sample_strategy=args.negative_sample_strategy, interact_times=new_node_val_data.node_interact_times)
    test_neg_edge_sampler = MultipleNegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2, negative_sample_strategy=args.negative_sample_strategy, interact_times=full_data.node_interact_times)
    new_node_test_neg_edge_sampler = MultipleNegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3, negative_sample_strategy=args.negative_sample_strategy, interact_times=new_node_test_data.node_interact_times)

    # get data loaders
    # Create data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    # Write example data to file
    with open('dataloader_examples.txt', 'w') as f:
        f.write("=== Example data from dataloaders ===\n\n")
        
        # Write training data examples
        f.write("Training data examples:\n")
        f.write(f"Number of training interactions: {len(train_data.src_node_ids)}\n")
        
        # Write first example with sample data
        f.write("\nExample 1:\n")
        f.write(f"Source node ID: {train_data.src_node_ids[0]}\n")
        f.write(f"Destination node ID: {train_data.dst_node_ids[0]}\n") 
        f.write(f"Interaction time: {train_data.node_interact_times[0]}\n")
        f.write(f"Edge ID: {train_data.edge_ids[0]}\n")
        f.write(f"Label: {train_data.labels[0]}\n")
        f.write(f"Index: {train_data.idx[0]}\n")
        f.write(f"Total number of interactions: {train_data.num_interactions}\n")
        f.write(f"Number of unique nodes: {train_data.num_unique_nodes}\n")
        f.write(f"Maximum source node ID: {train_data.src_max_id}\n")

        # Write batch info
        f.write(f"\nBatch size: {args.batch_size}\n")
        f.write(f"Number of batches in training: {len(train_idx_data_loader)}\n")

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        # args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        if args.model_name == 'GraphRec':
            dynamic_backbone = GraphRec(node_raw_features=node_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device, user_dynamic_features=user_dynamic_features, src_max_id=train_data.src_max_id)
        elif args.model_name == 'GraphRecMulti':
            dynamic_backbone = GraphRecMulti(node_raw_features=node_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device, user_dynamic_features=user_dynamic_features, src_max_id=train_data.src_max_id)
        elif args.model_name == 'GraphRecMultiCo':
            dynamic_backbone = GraphRecMultiCo(node_raw_features=node_raw_features, neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device, user_dynamic_features=user_dynamic_features,
                                         src_max_id=train_data.src_max_id, walk_length=args.walk_length, num_neighbors=args.num_neighbors)
        elif args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        subset_fraction = 0.1  # Train on 10% of the data
        num_batches = len(train_idx_data_loader)
        start_batch = int(num_batches * (1 - subset_fraction))  # Compute start index for last 90%

        train_loss_history, val_loss_history, new_val_loss_history = [], [], []
        train_acc_history, val_acc_history, new_val_acc_history = [], [], []
        train_pairwise_acc_history, val_pairwise_acc_history, new_val_pairwise_acc_history = [], [], []

        for epoch in range(args.num_epochs):

            model.train()
            if args.model_name in ['GraphRec', 'TGAT', 'GraphRecMulti', 'GraphRecMultiCo']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                if batch_idx < start_batch:
                    continue  # Skip first 90% of batches
                
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                # For dynamic features
                batch_src_idx = train_data.idx[train_data_indices]

                # # Save sample data to file for inspection
                # if batch_idx == start_batch:  # Only save first batch
                #     # Create table with headers and data
                #     headers = ['src_node', 'dst_node', 'timestamp', 'edge_id', 'src_idx']
                #     data = np.column_stack((
                #         batch_src_node_ids[:5],
                #         batch_dst_node_ids[:5], 
                #         batch_node_interact_times[:5],
                #         batch_edge_ids[:5],
                #         batch_src_idx[:5]
                #     ))
                    
                #     with open('sample_data.txt', 'w') as f:
                #         # Write headers
                #         f.write('\t'.join(headers) + '\n')
                #         # Write data rows
                #         np.savetxt(f, data, fmt='%d', delimiter='\t')
                #         f.write('\nhead=5\n\n')
                #         f.write('sample_source_indices:\n')
                #         f.write(str(batch_src_idx[:10]) + '\n\n')

                # batch_neg_dst_node_ids.shape: (batch_size, 4)
                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids), current_batch_start_time=batch_node_interact_times)

                # Log negative samples info
                with open('sample_negative_data.txt', 'a') as f:
                    f.write(f' batch_neg_dst_node_ids.shape: {batch_neg_dst_node_ids.shape}\n')
                    f.write('batch_neg_dst_node_ids: ')
                    f.write(str(batch_neg_dst_node_ids[:5]) + '\n')

                # batch_neg_src_node_ids = batch_src_node_ids
                batch_neg_src_node_ids = np.repeat(batch_src_node_ids, 4, axis=0).reshape(len(batch_src_node_ids), 4)
                batch_neg_src_idx = np.repeat(batch_src_idx, 4, axis=0).reshape(len(batch_src_idx), 4)

                # Flatten negative samples to compute embeddings properly
                batch_neg_src_node_ids_flat = batch_neg_src_node_ids.flatten()  # (batch_size * 4,)
                batch_neg_dst_node_ids_flat = batch_neg_dst_node_ids.flatten()  # (batch_size * 4,)
                batch_neg_times_flat = np.repeat(batch_node_interact_times, 4, axis=0).flatten()  # (batch_size * 4,)
                batch_neg_src_idx_flat = batch_neg_src_idx.flatten()  # (batch_size * 4,)

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['GraphRec', 'GraphRecMulti', 'GraphRecMultiCo']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          batch_src_idx=batch_src_idx)

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
                elif args.model_name in ['TGAT']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids_flat,
                                                                          dst_node_ids=batch_neg_dst_node_ids_flat,
                                                                          node_interact_times=batch_neg_times_flat,
                                                                          num_neighbors=args.num_neighbors)

                    # Reshape back to (batch_size, 4, node_feat_dim) so that each positive has 4 negatives
                    node_feat_dim = batch_neg_src_node_embeddings.shape[1]  # Get feature dimension
                    batch_neg_src_node_embeddings = batch_neg_src_node_embeddings.reshape(len(batch_src_node_ids), 4, node_feat_dim)
                    batch_neg_dst_node_embeddings = batch_neg_dst_node_embeddings.reshape(len(batch_src_node_ids), 4, node_feat_dim)

                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                # get positive and negative probabilities, shape (batch_size, )
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
                
                # Store loss
                train_losses.append(bpr_loss.item())

                predicts = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)  # (batch_size, 5)
                labels = torch.cat([torch.ones(positive_scores.shape[0], 1), torch.zeros(positive_scores.shape[0], 4)], dim=1)
                train_metrics.append(get_link_prediction_metrics(predicts, labels))
                
                # Optimize
                optimizer.zero_grad()
                bpr_loss.backward()
                optimizer.step()    

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {bpr_loss.item()}')

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     num_neighbors=args.num_neighbors)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       num_neighbors=args.num_neighbors)
            # Saving training results
            train_loss_history.append(np.mean(train_losses))
            val_loss_history.append(np.mean(val_losses))
            new_val_loss_history.append(np.mean(new_node_val_losses))
            train_acc_history.append(np.mean([train_metric['accuracy'] for train_metric in train_metrics]))
            val_acc_history.append(np.mean([val_metric['accuracy'] for val_metric in val_metrics]))
            new_val_acc_history.append(np.mean([new_node_val_metric['accuracy'] for new_node_val_metric in new_node_val_metrics]))
            train_pairwise_acc_history.append(np.mean([train_metric['pairwise_acc'] for train_metric in train_metrics]))
            val_pairwise_acc_history.append(np.mean([val_metric['pairwise_acc'] for val_metric in val_metrics]))
            new_val_pairwise_acc_history.append(np.mean([new_node_val_metric['pairwise_acc'] for new_node_val_metric in new_node_val_metrics]))

            # Save data as JSON
            training_results = {
                "train_loss_history": train_loss_history,
                "val_loss_history": val_loss_history,
                "new_val_loss_history": new_val_loss_history,
                "train_acc_history": train_acc_history,
                "val_acc_history": val_acc_history,
                "new_val_acc_history": new_val_acc_history,
                "train_pairwise_acc_history": train_pairwise_acc_history,
                "val_pairwise_acc_history": val_pairwise_acc_history,
                "new_val_pairwise_acc_history": new_val_pairwise_acc_history,
            }
            
            with open(f"saved_results/{args.model_name}/bluesky/training_results_{args.seed}.json", "w") as f:
                json.dump(training_results, f, indent=4)

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           num_neighbors=args.num_neighbors)

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             num_neighbors=args.num_neighbors)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                 model=model,
                                                                 neighbor_sampler=full_neighbor_sampler,
                                                                 evaluate_idx_data_loader=val_idx_data_loader,
                                                                 evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                 evaluate_data=val_data,
                                                                 num_neighbors=args.num_neighbors)

        new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                   model=model,
                                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                                   evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                   evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                   evaluate_data=new_node_val_data,
                                                                                   num_neighbors=args.num_neighbors)

        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   num_neighbors=args.num_neighbors)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     num_neighbors=args.num_neighbors)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
        for metric_name in new_node_val_metrics[0].keys():
            average_new_node_val_metric = np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics])
            logger.info(f'new node validate {metric_name}, {average_new_node_val_metric:.4f}')
            new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        val_metric_all_runs.append(val_metric_dict)
        new_node_val_metric_all_runs.append(new_node_val_metric_dict)

        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # Save data as JSON
        training_results = {
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "new_val_loss_history": new_val_loss_history,
            "train_acc_history": train_acc_history,
            "val_acc_history": val_acc_history,
            "new_val_acc_history": new_val_acc_history,
            "train_pairwise_acc_history": train_pairwise_acc_history,
            "val_pairwise_acc_history": val_pairwise_acc_history,
            "new_val_pairwise_acc_history": new_val_pairwise_acc_history,
        }
        
        with open(f"saved_results/{args.model_name}/bluesky/training_results_{args.seed}.json", "w") as f:
            json.dump(training_results, f, indent=4)

        # Save plots using the function
        save_plot(
            [train_loss_history, val_loss_history, new_val_loss_history],
            ["Train Loss", "Validation Loss", "New Node Validation Loss"],
            "Training and Validation Loss",
            "Loss",
            f"saved_results/{args.model_name}/bluesky/training_loss_plot_{args.seed}.png",
        )
        
        save_plot(
            [train_acc_history, val_acc_history, new_val_acc_history],
            ["Train Accuracy", "Validation Accuracy", "New Node Validation Accuracy"],
            "Training and Validation Accuracy",
            "Accuracy",
            f"saved_results/{args.model_name}/bluesky/training_accuracy_plot_{args.seed}.png",
        )
        
        save_plot(
            [train_pairwise_acc_history, val_pairwise_acc_history, new_val_pairwise_acc_history],
            ["Train Pairwise Accuracy", "Validation Pairwise Accuracy", "New Node Validation Pairwise Accuracy"],
            "Training and Validation Pairwise Accuracy",
            "Pairwise Accuracy",
            f"saved_results/{args.model_name}/bluesky/training_pairwise_accuracy_plot_{args.seed}.png",
        )

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]:.4f}' for metric_name in val_metric_dict},
            "new node validate metrics": {metric_name: f'{new_node_val_metric_dict[metric_name]:.4f}' for metric_name in new_node_val_metric_dict},
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)
        

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(f'validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}')
        logger.info(f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                    f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_val_metric_all_runs[0].keys():
        logger.info(f'new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}')
        logger.info(f'average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
