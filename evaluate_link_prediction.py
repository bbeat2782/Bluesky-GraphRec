import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch.nn as nn
import pickle
import pandas as pd

from models.TGAT import TGAT
from models.GraphRecMultiCo import GraphRecMultiCo
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, CandidateEdgeSampler
from evaluate_models_utils import evaluate_real
# from evaluate_models_utils_candidates import evaluate_real
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data, get_link_prediction_data_eval
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.candidates import EmbeddingCandidateEdgeSampler

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # Suppress Matplotlib debug messages once at the beginning
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    # get arguments
    args = get_link_prediction_args(is_evaluation=True)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, test_data, eval_test_data, dynamic_user_features, post_dynamic_features = \
        get_link_prediction_data_eval(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    
    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    if args.negative_sample_strategy == 'real':  
        # new_node_test_neg_edge_sampler = CandidateEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, interact_times=full_data.node_interact_times)     
        # Load post embeddings from pickle file which is in list format
        dynamic_user_features_path = './processed_data/bluesky/user_dynamic_features.pkl' # changed from DG_data
        with open(dynamic_user_features_path, "rb") as file:
            dynamic_user_features = pickle.load(file)

        post_embeddings_path = os.path.join(os.path.expanduser("~"), 'post_dynamic_embeddings.parquet')
        post_embeddings_df = pd.read_parquet(post_embeddings_path)
        if 'prev_embedding' in post_embeddings_df.columns:
            post_embeddings_df = post_embeddings_df.drop(columns=['prev_embedding'])
        
        # Initialize with embedding-based candidate sampler instead of heuristic sampler
        new_node_test_neg_edge_sampler = EmbeddingCandidateEdgeSampler(
            user_dynamic_features=dynamic_user_features,
            post_embeddings_df=post_embeddings_df,
            time_window_hours=args.time_window_hours if hasattr(args, 'time_window_hours') else 24,
            n_candidates=args.n_candidates if hasattr(args, 'n_candidates') else 1000,
            seed=args.seed
        )    
    else:
        raise ValueError(f'negative sample strategy should be `real`. It is {args.negative_sample_strategy}.')
    
    # get data loaders
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(eval_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    result_json = {}

    for run in range(args.num_runs):
        set_random_seed(seed=run)

        # args.seed = run
        args.load_model_name = f'{args.model_name}_seed{args.seed}_{run+1}'
        args.save_result_name = f'{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{str(time.time())}.log")
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
        if args.model_name == 'GraphRecMultiCo':
            dynamic_backbone = GraphRecMultiCo(node_raw_features=node_raw_features, neighbor_sampler=full_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                            num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                            max_input_sequence_length=args.max_input_sequence_length, device=args.device, user_dynamic_features=dynamic_user_features,
                                            post_dynamic_features=post_dynamic_features,
                                            src_max_id=eval_test_data.src_max_id, walk_length=args.walk_length, num_neighbors=args.num_neighbors)

            link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1]+64, input_dim2=node_raw_features.shape[1]+64,
                                    hidden_dim=node_raw_features.shape[1]+64, output_dim=1)
        elif args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=full_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
            link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # load the saved model
        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
        early_stopping = EarlyStopping(patience=0, save_model_folder=load_model_folder,
                                        save_model_name=args.load_model_name, logger=logger, model_name=args.model_name)
        early_stopping.load_checkpoint(model, map_location='cpu')

        model = convert_to_gpu(model, device=args.device)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        avg_mrr, avg_ild = evaluate_real(model_name=args.model_name,
                                        model=model,
                                        neighbor_sampler=full_neighbor_sampler,
                                        evaluate_idx_data_loader=test_idx_data_loader,
                                        evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                        evaluate_data=eval_test_data,
                                        num_neighbors=args.num_neighbors,
                                        time_gap=args.time_gap,
                                        load_model_name=args.load_model_name,
                                        src_node_ids=full_data.src_node_ids,
                                        dst_node_ids=full_data.dst_node_ids,
                                        interact_times=full_data.node_interact_times
                                        )

        logger.info(f'Test MRR, {avg_mrr:.4f}')
        logger.info(f'Test ILD@10, {avg_ild:.4f}')
    
        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json[f"new node test metrics_{run+1}"] = {'MRR': avg_mrr, 'ILD@10': avg_ild}

    mrr_values = []
    ild_values = []
    for key in result_json:
        # Convert the MRR value to a float (if necessary)
        mrr_values.append(float(result_json[key]['MRR']))
        ild_values.append(float(result_json[key]['ILD@10']))
    
    print("Mean MRR:", np.mean(mrr_values))
    print("Std Dev MRR:", np.std(mrr_values))

    print("Mean ILD@10:", np.mean(ild_values))
    print("Std Dev ILD@10:", np.std(ild_values))
    
    result_json = json.dumps(result_json, indent=4)

    save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
    os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, f"{args.save_result_name}.json")
    with open(save_result_path, 'w') as file:
        file.write(result_json)
    logger.info(f'save negative sampling results at {save_result_path}')

    sys.exit()
