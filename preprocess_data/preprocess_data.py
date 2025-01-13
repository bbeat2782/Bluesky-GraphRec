# modifying for bluesky dataset
# Item node feature:
#    currently, text embedding has 384 dim --> which is item node feature (not edge feature)
# User node feataure:
#    work in progress: concatenate a bunch of features?
# Edge feature:
#    comment text embedding? --> but then, how do we use this during inference?
# Time encoding:
#    use Time Encoder from models/modules/TimeEncoder
# Neighbor encoding:
#    work in progress

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from pandas.testing import assert_frame_equal
from distutils.dir_util import copy_tree


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
    ITEM_EMBEDDINGS_PATH = '../DG_data/{}/{}_text_embeddings.csv'.format(dataset_name, dataset_name)
    OUT_DF = '../processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    OUT_FEAT = '../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    OUT_NODE_FEAT = '../processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)

    df, edge_feats = preprocess(INTERACTION_PATH)
    new_df = reindex(df, bipartite)


    # edge feature for zero index, which is not used (since edge id starts from 1)
    empty = np.zeros(edge_feats.shape[1])[np.newaxis, :]
    # Stack arrays in sequence vertically(row wise),
    edge_feats = np.vstack([empty, edge_feats])

    # Processing node features
    text_embeddings = pd.read_csv(ITEM_EMBEDDINGS_PATH)
    text_embeddings = text_embeddings.sort_values('item_id').reset_index(drop=True)
    user_max_id = df.u.max()
    offset = user_max_id + 2
    text_embeddings['item_id'] = text_embeddings['item_id'] + offset

    # node features with one additional feature for zero index (since node id starts from 1)
    max_idx = max(new_df.u.max(), new_df.i.max())
    node_feats = np.zeros((max_idx + 1, node_feat_dim))

    for _, row in text_embeddings.iterrows():
        global_item_id = row['item_id']
        embedding = np.array([float(x) for x in row['comma_separated_list_of_features'].split(',')], dtype=np.float16)  # Parse embedding
        node_feats[global_item_id] = embedding

    print('number of nodes ', node_feats.shape[0] - 1)
    print('number of node features ', node_feats.shape[1])
    print('number of edges ', edge_feats.shape[0] - 1)
    print('number of edge features ', edge_feats.shape[1])

    new_df.to_csv(OUT_DF)  # edge-list
    np.save(OUT_FEAT, edge_feats)  # edge features
    np.save(OUT_NODE_FEAT, node_feats)  # node features


def check_data(dataset_name: str):
    """
    check whether the processed datasets are identical to the given processed datasets
    :param dataset_name: str, dataset name
    :return:
    """
    # original data paths
    origin_OUT_DF = '../DG_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    origin_OUT_FEAT = '../DG_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    origin_OUT_NODE_FEAT = '../DG_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)

    # processed data paths
    OUT_DF = '../processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    OUT_FEAT = '../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    OUT_NODE_FEAT = '../processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)

    # Load original data
    origin_g_df = pd.read_csv(origin_OUT_DF)
    origin_e_feat = np.load(origin_OUT_FEAT)
    origin_n_feat = np.load(origin_OUT_NODE_FEAT)

    # Load processed data
    g_df = pd.read_csv(OUT_DF)
    e_feat = np.load(OUT_FEAT)
    n_feat = np.load(OUT_NODE_FEAT)

    assert_frame_equal(origin_g_df, g_df)
    # check numbers of edges and edge features
    assert origin_e_feat.shape == e_feat.shape and origin_e_feat.max() == e_feat.max() and origin_e_feat.min() == e_feat.min()
    # check numbers of nodes and node features
    assert origin_n_feat.shape == n_feat.shape and origin_n_feat.max() == n_feat.max() and origin_n_feat.min() == n_feat.min()


parser = argparse.ArgumentParser('Interface for preprocessing datasets')
parser.add_argument('--dataset_name', type=str,
                    choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'enron', 'SocialEvo', 'uci',
                             'Flights', 'CanParl', 'USLegis', 'UNtrade', 'UNvote', 'Contacts', 'bluesky'],
                    help='Dataset name', default='wikipedia')
parser.add_argument('--node_feat_dim', type=int, default=128, help='Number of node raw features')

args = parser.parse_args()

print(f'preprocess dataset {args.dataset_name}...')
if args.dataset_name in ['enron', 'SocialEvo', 'uci']:
    Path("../processed_data/{}/".format(args.dataset_name)).mkdir(parents=True, exist_ok=True)
    copy_tree("../DG_data/{}/".format(args.dataset_name), "../processed_data/{}/".format(args.dataset_name))
    print(f'the original dataset of {args.dataset_name} is unavailable, directly use the processed dataset by previous works.')
else:
    # bipartite dataset
    if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'myket', 'bluesky']:
        preprocess_data(dataset_name=args.dataset_name, bipartite=True, node_feat_dim=args.node_feat_dim)
    else:
        preprocess_data(dataset_name=args.dataset_name, bipartite=False, node_feat_dim=args.node_feat_dim)
    print(f'{args.dataset_name} is processed successfully.')

    if args.dataset_name not in ['myket']:
        check_data(args.dataset_name)
    print(f'{args.dataset_name} passes the checks successfully.')
