import pandas as pd
import numpy as np
import pickle

# Load data
dataset_name = 'bluesky'
graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

dynamic_user_features_path = './DG_data/bluesky/user_dynamic_features.pkl'
with open(dynamic_user_features_path, 'rb') as f:
    dynamic_user_features = pickle.load(f)

# Write data info to file
with open('data_info.txt', 'w') as f:
    f.write("=== Graph DataFrame ===\n")
    f.write(f"Shape: {graph_df.shape}\n")
    f.write("\nFirst 5 rows:\n")
    f.write(f"graph_df: ml_{dataset_name}.csv\n")
    f.write(str(graph_df.head()))

    f.write("\n\n=== Edge Features ===\n")
    f.write(f"Shape: {edge_raw_features.shape}\n")
    f.write(f"\nedge_raw_features: ml_{dataset_name}.npy\n")
    f.write("\nFirst 5 rows:\n")
    f.write(str(edge_raw_features[:5]))

    f.write("\n\n=== Node Features ===\n")
    f.write(f"Shape: {node_raw_features.shape}\n")
    f.write(f"\nnode_raw_features: ml_{dataset_name}_node.npy\n") 
    f.write("\nFirst 5 rows:\n")
    f.write(str(node_raw_features[:5]))
    
    f.write("\n\n=== Dynamic User Features ===\n")
    f.write(f"Number of users: {len(dynamic_user_features)}\n")
    f.write(f"\ndynamic_user_features: {dataset_name}/user_dynamic_features.pkl\n")
    f.write("\nFirst user:\n")
    items = list(dynamic_user_features.items())[:1]
    for user_id, features in items:
        f.write(f"User {user_id}: {features}\n")