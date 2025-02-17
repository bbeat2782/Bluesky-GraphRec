from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import pickle
from datetime import datetime


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, idx, src_max_id):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.idx = idx
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.src_max_id = src_max_id


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

    dynamic_user_features_path = '/home/sgan/private/DyGLib/DG_data/bluesky/user_dynamic_features.pkl'
    dynamic_user_features_path = './DG_data/bluesky/user_dynamic_features.pkl'
    with open(dynamic_user_features_path, "rb") as file:
        dynamic_user_features = pickle.load(file)

    # # Stage 1: Initial data load
    # with open('stage1_initial_data.txt', 'w') as f:
    #     f.write("=== Initial Raw Data ===\n")
    #     f.write("graph_df: ml_bluesky.csv\n")
    #     f.write(f"graph_df Shape: {graph_df.shape}\n")
    #     f.write(f"graph_df Head:\n{graph_df.head()}\n\n")
    #     f.write(f"edge_raw_features: ml_bluesky.npy\n")
    #     f.write(f"edge_raw_features Shape: {edge_raw_features.shape}\n")
    #     f.write(f"edge_raw_features First 5 rows:\n{edge_raw_features[:5]}\n\n")
    #     f.write(f"node_raw_features: ml_bluesky_node.npy\n")
    #     f.write(f"node_raw_features Shape: {node_raw_features.shape}\n")
    #     f.write(f"node_raw_features First 5 rows:\n{node_raw_features[:5]}\n\n")
    #     f.write(f"dynamic_user_features: bluesky/user_dynamic_features.pkl\n")
    #     f.write(f"dynamic_user_features Shape: {len(dynamic_user_features)}\n")
    #     items_list = list(dynamic_user_features.items())[:20]
    #     f.write(f"dynamic_user_features First 20 rows:\n{items_list}\n\n")

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 128

    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    
    # padding the features
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    # # Stage 2: After padding
    # with open('stage2_after_padding.txt', 'w') as f:
    #     f.write("=== After Feature Padding ===\n")
    #     f.write(f"node_raw_features Shape: {node_raw_features.shape}\n")
    #     f.write(f"node_raw_features First 5 rows:\n{node_raw_features[:5]}\n\n")
    #     f.write(f"edge_raw_features Shape: {edge_raw_features.shape}\n")
    #     f.write(f"edge_raw_features First 5 rows:\n{edge_raw_features[:5]}\n")

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'
    
    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    print('val_time:', datetime.utcfromtimestamp(val_time).strftime('%Y-%m-%d %H:%M:%S'))
    print('test_time:', datetime.utcfromtimestamp(test_time).strftime('%Y-%m-%d %H:%M:%S'))
  
    src_node_ids = graph_df.u.values.astype(np.int32)
    dst_node_ids = graph_df.i.values.astype(np.int32)
    edge_ids = graph_df.idx.values.astype(np.int32)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    labels = graph_df.label.values.astype(np.int8)
    idx = graph_df.idx.values.astype(np.int32)
    src_max_id = np.max(src_node_ids)

    # # Stage 3: After array extraction
    # with open('stage3_extracted_arrays.txt', 'w') as f:
    #     f.write("=== Extracted Arrays ===\n")
    #     f.write(f"Split Times - Val: {val_time}, Test: {test_time}\n\n")
    #     f.write(f"src_node_ids (shape: {src_node_ids.shape}):\n{src_node_ids[:20]}\n\n")
    #     f.write(f"dst_node_ids (shape: {dst_node_ids.shape}):\n{dst_node_ids[:20]}\n\n")
    #     f.write(f"edge_ids (shape: {edge_ids.shape}):\n{edge_ids[:20]}\n\n")
    #     f.write(f"node_interact_times (shape: {node_interact_times.shape}):\n{node_interact_times[:20]}\n\n")
    #     f.write(f"labels (shape: {labels.shape}):\n{labels[:20]}\n\n")
    #     f.write(f"src_max_id: {src_max_id}\n")

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, idx=idx, src_max_id=src_max_id)

    random.seed(2020)

    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(
        set(dst_node_ids[node_interact_times > val_time])
    )
    sorted_test_node_list = sorted(test_node_set)
    new_test_node_set = set(random.sample(sorted_test_node_list, int(0.1 * num_total_unique_node_ids)))

    # # Stage 4: After node set creation
    # with open('stage4_node_sets.txt', 'w') as f:
    #     f.write("=== Node Sets ===\n")
    #     f.write(f"num_total_unique_node_ids: {num_total_unique_node_ids}\n")
    #     f.write(f"len(test_node_set): {len(test_node_set)}\n")
    #     f.write(f"Sorted Sample of Test Node Set (first 20):\n{sorted(list(test_node_set))[:20]}\n\n")
    #     f.write(f"len(new_test_node_set): {len(new_test_node_set)}\n")
    #     f.write(f"Sorted Sample of New Test Node Set (first 20):\n{sorted(list(new_test_node_set))[:20]}\n")

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask], idx=idx[train_mask], src_max_id=src_max_id)

    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], 
                    labels=labels[val_mask], idx=idx[val_mask], src_max_id=src_max_id)

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], 
                     labels=labels[test_mask], idx=idx[test_mask], src_max_id=src_max_id)

    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask], 
                             idx=idx[new_node_val_mask], src_max_id=src_max_id)

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask], 
                              idx=idx[new_node_test_mask], src_max_id=src_max_id)

    # # Stage 5: Final splits
    # with open('stage5_final_splits.txt', 'w') as f:
    #     f.write("=== Final Dataset Splits ===\n")
    #     for name, data in [
    #         ("Full", full_data),
    #         ("Train", train_data), 
    #         ("Val", val_data),
    #         ("Test", test_data),
    #         ("New Node Val", new_node_val_data),
    #         ("New Node Test", new_node_test_data)
    #     ]:
    #         f.write(f"\n{name} Dataset:\n")
    #         f.write(f"num_interactions: {data.num_interactions}\n")
    #         f.write(f"num_unique_nodes: {data.num_unique_nodes}\n")
    #         f.write(f"First 5 src_node_ids: {data.src_node_ids[:5]}\n")
    #         f.write(f"First 5 dst_node_ids: {data.dst_node_ids[:5]}\n")
    #         f.write(f"First 5 node_interact_times: {data.node_interact_times[:5]}\n")
    #         f.write(f"First 5 labels: {data.labels[:5]}\n\n")

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, dynamic_user_features


def get_link_prediction_data_eval(dataset_name: str, val_ratio: float, test_ratio: float):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))
    dynamic_user_features_path = './DG_data/bluesky/user_dynamic_features.pkl'
    with open(dynamic_user_features_path, "rb") as file:
        dynamic_user_features = pickle.load(file)


    NODE_FEAT_DIM = EDGE_FEAT_DIM = 128

    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    print('val_time:', datetime.utcfromtimestamp(val_time).strftime('%Y-%m-%d %H:%M:%S'))
    print('test_time:', datetime.utcfromtimestamp(test_time).strftime('%Y-%m-%d %H:%M:%S'))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    idx = graph_df.idx.values
    src_max_id = np.max(src_node_ids)

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, idx=idx, src_max_id=src_max_id)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    # Convert the set to a sorted list
    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(
        set(dst_node_ids[node_interact_times > val_time])
    )
    sorted_test_node_list = sorted(test_node_set)
    # Sample nodes from the sorted list
    new_test_node_set = set(random.sample(sorted_test_node_list, int(0.1 * num_total_unique_node_ids)))

    # mask for each source and destination to denote whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # mask, which is true for edges with both destination and source not being new test nodes (because we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # for train data, we keep edges happening before the validation time which do not involve any new node, used for inductiveness
    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask], idx=idx[train_mask], src_max_id=src_max_id)

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    # new nodes that are not in the training set
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    test_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                     node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask], idx=idx[val_mask], src_max_id=src_max_id)

    # Filter relevant data for new node test
    filtered_src_node_ids = src_node_ids[val_mask]
    filtered_dst_node_ids = dst_node_ids[val_mask]
    filtered_node_interact_times = node_interact_times[val_mask]
    filtered_edge_ids = edge_ids[val_mask]
    filtered_labels = labels[val_mask]
    filtered_idx = idx[val_mask]

    
    length_restrict = int(0.001 * len(filtered_src_node_ids))

    interactions_df = pd.DataFrame({
        'src_node_id': filtered_src_node_ids[:length_restrict],
        'dst_node_id': filtered_dst_node_ids[:length_restrict],
        'node_interact_time': filtered_node_interact_times[:length_restrict],
        'edge_id': filtered_edge_ids[:length_restrict],
        'label': filtered_labels[:length_restrict],
        'idx': filtered_idx[:length_restrict]
    })

    # Sort by src_node_id and node_interact_time
    interactions_df = interactions_df.sort_values(by=['src_node_id', 'node_interact_time'])
    
    # Extract the last interaction for each src_node_id
    last_interactions = interactions_df.groupby('src_node_id').last().reset_index()

    # Construct eval_test_data from last_interactions
    eval_test_data = Data(
        src_node_ids=last_interactions['src_node_id'].values,
        dst_node_ids=last_interactions['dst_node_id'].values,
        node_interact_times=last_interactions['node_interact_time'].values,
        edge_ids=last_interactions['edge_id'].values,
        labels=last_interactions['label'].values,
        idx=last_interactions['idx'].values,
        src_max_id=src_max_id
    )

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
        eval_test_data.num_interactions, eval_test_data.num_unique_nodes))
    print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(len(new_test_node_set)))

    return node_raw_features, edge_raw_features, full_data, test_data, eval_test_data, dynamic_user_features
