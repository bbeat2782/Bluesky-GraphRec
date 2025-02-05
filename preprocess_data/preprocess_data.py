import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from pandas.testing import assert_frame_equal
from distutils.dir_util import copy_tree
from datetime import datetime, timedelta
import pickle
import time


def preprocess(dataset_name: str):
    """
    read the original data file and return the DataFrame that has columns ['u', 'i', 'ts', 'label', 'idx']
    :param dataset_name: str, dataset name
    :return:
    """    
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(dataset_name) as f:
        # skip the first line
        s = next(f)
        previous_time = -1
        for idx, line in enumerate(f):
            e = line.strip().split(',')
            # user_id
            u = int(e[0])
            # item_id
            i = int(e[1])

            # timestamp
            ts = float(e[2])
            # check whether time in ascending order
            # TODO : Check if we can get second from like history
            assert ts >= previous_time
            previous_time = ts
            # state_label
            label = float(e[3])

            # edge features --> dataset_name.csv does not have e[4:]
            # feat = np.array([float(x) for x in e[4:]])
            feat = np.array([0.0, 0.0])  # dim 2 with zeros

            #feat = item_embeddings.get(i, np.zeros_like(list(item_embeddings.values())[0]))  # Default to zeros if missing

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            # edge index
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


def reindex(df: pd.DataFrame, bipartite: bool = True):
    """
    reindex the ids of nodes and edges
    :param df: DataFrame
    :param bipartite: boolean, whether the graph is bipartite or not
    :return:
    """
    new_df = df.copy()
    if bipartite:
        # check the ids of users and items
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))
        assert df.u.min() == df.i.min() == 0

        # if bipartite, discriminate the source and target node by unique ids (target node id is counted based on source node id)
        upper_u = df.u.max() + 1
        new_i = df.i + upper_u

        new_df.i = new_i

    # make the id start from 1
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    return new_df

def unpack_embeddings(packed_bytes):
    # Convert binary blob back to array and unpack bits
    return np.unpackbits(np.frombuffer(packed_bytes, dtype=np.uint8))[:128]

def preprocess_data(dataset_name: str, bipartite: bool = True, node_feat_dim: int = 128, edge_feat_dim: int = 10):
    """
    preprocess the data
    :param dataset_name: str, dataset name
    :param bipartite: boolean, whether the graph is bipartite or not
    :param node_feat_dim: int, dimension of node features
    :return:
    """
    Path("../processed_data/{}/".format(dataset_name)).mkdir(parents=True, exist_ok=True)
    INTERACTION_PATH = '../DG_data/{}/{}.csv'.format(dataset_name, dataset_name)
    ITEM_EMBEDDINGS_PATH = '../DG_data/{}/{}_text_embeddings.parquet'.format(dataset_name, dataset_name)
    OUT_DF = '../processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    OUT_FEAT = '../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    OUT_NODE_FEAT = '../processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)
    OUT_DYNAMIC_USER_FEAT = '../processed_data/{}/ml_{}_user_dynamic.npy'.format(dataset_name, dataset_name)

    df, edge_feats = preprocess(INTERACTION_PATH)
    new_df = reindex(df, bipartite)

    # edge feature for zero index, which is not used (since edge id starts from 1)
    empty = np.zeros(edge_feats.shape[1])[np.newaxis, :]
    # Stack arrays in sequence vertically(row wise),
    edge_feats = np.vstack([empty, edge_feats])
    
    # Processing node features from Parquet file
    text_embeddings = pd.read_parquet(ITEM_EMBEDDINGS_PATH)
    text_embeddings = text_embeddings.sort_values('item_id').reset_index(drop=True)
    user_max_id = df.u.max()
    offset = user_max_id + 2
    text_embeddings['item_id'] = text_embeddings['item_id'] + offset

    # node features with one additional feature for zero index (since node id starts from 1)
    max_idx = max(new_df.u.max(), new_df.i.max())
    node_feats = np.zeros((max_idx + 1, node_feat_dim))

    # storing post embeddings
    for _, row in text_embeddings.iterrows():
        global_item_id = row['item_id']
        node_feats[global_item_id] = row['embeddings']
        
    # start_time = time.time()
    # # Current format of ts: `20230101024321.0` (`YYYYMMDDHHMMSS`) --> change to datetime obj
    # new_df['ts'] = pd.to_datetime(new_df['ts'].astype(int).astype(str), format='%Y%m%d%H%M%S')

    # history_size = 10

    # # Ensure dataframe is sorted
    # new_df = new_df.sort_values(by=['u', 'ts']).reset_index(drop=True)
    
    # # Convert to NumPy for fast access
    # user_ids = new_df['u'].values
    # item_ids = new_df['i'].values
    # embeddings = node_feats[item_ids].astype(np.float16)  # Direct NumPy lookup
    
    # # Initialize output array
    # user_dynamic_features = np.zeros_like(embeddings, dtype=np.float16)
    
    # # Process in batch using NumPy slicing
    # unique_users, user_starts = np.unique(user_ids, return_index=True)
    # for idx, start in enumerate(user_starts):
    #     end = user_starts[idx + 1] if idx + 1 < len(user_starts) else len(user_ids)
        
    #     user_embeds = embeddings[start:end]  # Extract all embeddings for this user
    #     num_interactions = len(user_embeds)
    
    #     if num_interactions == 1:
    #         # If only one interaction exists, set to zeros (no past interactions to average)
    #         user_dynamic_features[start:end] = np.zeros_like(user_embeds)
    #         continue
    
    #     # Compute cumulative sum
    #     cumsum = np.cumsum(user_embeds, axis=0)
    
    #     # Compute rolling sum while excluding current embedding
    #     rolling_sum = np.zeros_like(user_embeds)
    #     for i in range(num_interactions):
    #         start_idx = max(0, i - history_size)
    #         past_sum = cumsum[i - 1] - (cumsum[start_idx - 1] if start_idx > 0 else 0)
    #         rolling_sum[i] = past_sum
    
    #     # Compute rolling mean excluding the current embedding
    #     valid_counts = np.minimum(np.arange(num_interactions), history_size)[:, None]  # Excludes current element
    #     rolling_mean = rolling_sum / np.maximum(valid_counts, 1)  # Avoid division by zero
    
    #     # Explicitly set the first interaction to a zero vector
    #     rolling_mean[0] = np.zeros_like(user_embeds[0])
    
    #     # Store result
    #     user_dynamic_features[start:end] = rolling_mean
    
    # # Restore the original order before saving
    # new_df['user_dynamic_features'] = list(user_dynamic_features)
    # new_df = new_df.sort_values(by='idx')

    # np.savez_compressed("/home/sgan/user_dynamic_features.npz", 
    #                 user_dynamic_features=np.array(new_df['user_dynamic_features'].tolist(), dtype=np.float16))

    
    print('number of nodes ', node_feats.shape[0] - 1)
    print('number of node features ', node_feats.shape[1])
    print('number of edges ', edge_feats.shape[0] - 1)
    print('number of edge features ', edge_feats.shape[1])
    # print('number of dynamic features ', user_dynamic_features.shape[1])
    
    new_df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features
    # np.save(OUT_DYNAMIC_USER_FEAT, user_dynamic_features)  # dynamic features
    

parser = argparse.ArgumentParser('Interface for preprocessing datasets')
parser.add_argument('--dataset_name', type=str,
                    choices=['bluesky'],
                    help='Dataset name', default='bluesky')
parser.add_argument('--node_feat_dim', type=int, default=128, help='Number of node raw features')

args = parser.parse_args()

print(f'preprocess dataset {args.dataset_name}...')

# bipartite dataset
if args.dataset_name in ['bluesky']:
    preprocess_data(dataset_name=args.dataset_name, bipartite=True, node_feat_dim=args.node_feat_dim)
else:
    preprocess_data(dataset_name=args.dataset_name, bipartite=False, node_feat_dim=args.node_feat_dim)
print(f'{args.dataset_name} is processed successfully.')

print(f'{args.dataset_name} passes the checks successfully.')
