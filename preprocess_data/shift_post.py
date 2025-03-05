import pandas as pd
import os
import numpy as np

dataset_name = 'bluesky'
graph_df = pd.read_csv('../processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))

post_embeddings_path = os.path.join(os.path.expanduser("~"), 'post_dynamic_embeddings.parquet')
post_embeddings = pd.read_parquet(post_embeddings_path)

# Convert 'timestamp' column to datetime (if it isn't already)
post_embeddings['timestamp'] = pd.to_datetime(post_embeddings['timestamp'])

# Convert to Unix time (seconds since epoch)
post_embeddings['timestamp'] = post_embeddings['timestamp'].astype('int64') // 10**9

post_embeddings = post_embeddings.sort_values(by=['post_id', 'timestamp']).reset_index(drop=True)
post_embeddings = post_embeddings.drop_duplicates(subset=['user_id', 'post_id', 'timestamp'], keep='first')

# Perform left merge to preserve bluesky_data order
merged_df = graph_df.merge(
    post_embeddings,
    how='left',
    left_on=['u', 'i', 'ts'],  # bluesky_data columns
    right_on=['user_id', 'post_id', 'timestamp']  # post_dynamic_df columns
)

# Drop redundant columns if needed
merged_df = merged_df.drop(columns=['user_id', 'post_id'])

assert len(merged_df) == len(graph_df)

# Fill missing values with a NumPy array of 64 zeros (dtype=float32 for efficiency)
merged_df['prev_embedding'] = merged_df['prev_embedding'].apply(
    lambda x: x if isinstance(x, (list, np.ndarray)) else np.zeros(64, dtype=np.float16)
)

# Ensure all values are NumPy arrays for consistency
merged_df['prev_embedding'] = merged_df['prev_embedding'].apply(
    lambda x: np.array(x, dtype=np.float16)
)

# Convert shifted_embeddings column to a NumPy array with float16 precision
final_embeddings = np.stack(merged_df['prev_embedding'].values).astype(np.float16)

# Save to file
shifted_post_embeddings_path = os.path.join(os.path.expanduser("~"), 'post_dynamic_embeddings_shifted.npy')
np.save(shifted_post_embeddings_path, final_embeddings)

# Verify saved shape
print("Saved embeddings shape:", final_embeddings.shape)