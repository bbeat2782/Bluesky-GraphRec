import duckdb
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import time
from tqdm import tqdm
import torch

def pack_embeddings(binary_values):
    return np.packbits(binary_values).tobytes()

con = duckdb.connect('../random_tests/scan_results.duckdb')

# Get the data
df = con.execute("""
    SELECT 
        TRIM(
            LEADING '/' FROM 
            TRIM(
                TRAILING '"' FROM 
                CONCAT(
                    REPLACE(SUBSTR(json_extract(record, '$.subject.uri')::STRING, 6), '/app.bsky.feed.post/', '_')
                )
            ) 
        ) AS formatted_postUri,
        repo AS userId,
        createdAt
    FROM 
        records
    WHERE 
        createdAt >= '2023-01-01' 
        AND collection = 'app.bsky.feed.like'
""").fetchdf()

# Create a mapping for sequential numbering for posts
unique_posts = df['formatted_postUri'].unique()
post_mapping = {post: i for i, post in enumerate(unique_posts)}
# Remap `formatted_postUri` to sequential numbers
df['formatted_postUri'] = df['formatted_postUri'].map(post_mapping)

# Create a mapping for sequential numbering for users
unique_users = df['userId'].unique()
user_mapping = dict(zip(unique_users, np.arange(len(unique_users))))
df['userId'] = df['userId'].map(user_mapping)

# changing df to match the format that the preprocessing code takes
df = df.rename(columns={
    'userId': 'source_node',
    'formatted_postUri': 'destination_node',
    'createdAt': 'timestamp'})
df['edge_label'] = 0
df = df[['source_node', 'destination_node', 'timestamp', 'edge_label']]
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y%m%d%H%M%S')
df = df.sort_values(by='timestamp')

df.to_csv('../DG_data/bluesky/bluesky.csv', index=False)
with open('../DG_data/bluesky/post_mapping.pkl', 'wb') as f:
    pickle.dump(post_mapping, f)
with open('../DG_data/bluesky/user_mapping.pkl', 'wb') as f:
    pickle.dump(user_mapping, f)
    

# Get all post data
post_df = con.execute("""
    SELECT createdAt, repo, rkey, json_extract(record, '$.text') AS text
    FROM records
    WHERE createdAt >= '2023-01-01' AND collection == 'app.bsky.feed.post'
    ORDER BY createdAt ASC
""").fetchdf()

post_df['key'] = post_df['repo'] + '_' + post_df['rkey']
post_df['key'] = post_df['key'].map(post_mapping)
post_df = post_df[~post_df['key'].isna()]
post_df['key'] = post_df['key'].astype(int)
post_df = post_df[['key', 'text']]
post_df = post_df.rename(columns={'key': 'item_id'})

# Text embedding
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(device)
batch_size = 1024 * 4
texts = post_df['text'].tolist()

# Process posts in batches
all_embeddings = []
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i + batch_size]
    with torch.no_grad():
        embeddings = model.encode(batch_texts, truncate_dim=128)  # Ensure model compatibility
        embeddings = embeddings.astype(np.float16)
        #binary_embeddings = (embeddings > 0).astype(np.uint8)  # Efficient binary transformation
        all_embeddings.extend(embeddings.tolist())

# Store post embeddings
post_df = post_df.drop(columns=['text'])
post_df['embeddings'] = all_embeddings
parquet_file_path = "/home/sgan/private/DyGLib/DG_data/bluesky/bluesky_text_embeddings.parquet"
#post_df.to_parquet(parquet_file_path, index=False, compression='zstd')
post_df.to_parquet(parquet_file_path, index=False, compression='zstd', engine='pyarrow')


# Get user-post relationships
user_posts = con.execute("""
    SELECT repo AS userId, CONCAT(repo, '_', rkey) AS post_key
    FROM records
    WHERE createdAt >= '2023-01-01' AND collection == 'app.bsky.feed.post'
""").fetchdf()

# Map post keys to sequential IDs
user_posts['post_key'] = user_posts['post_key'].map(post_mapping)
user_posts = user_posts[~user_posts['post_key'].isna()]
user_posts['userId'] = user_posts['userId'].map(user_mapping)

# Calculate average embeddings for each user
user_embeddings = []
for user_id in tqdm(range(len(user_mapping))):
    user_post_ids = user_posts[user_posts['userId'] == user_id]['post_key'].values
    if len(user_post_ids) > 0:
        user_post_embeddings = post_df[post_df['item_id'].isin(user_post_ids)]['embeddings'].values
        avg_embedding = np.mean(user_post_embeddings, axis=0)
    else:
        avg_embedding = np.zeros(128)  # Default embedding for users with no posts
    user_embeddings.append(avg_embedding)

# Create user embeddings dataframe
user_df = pd.DataFrame({
    'user_id': range(len(user_mapping)),
    'embeddings': user_embeddings
})

user_df.to_parquet("../DG_data/bluesky/bluesky_user_embeddings.parquet", index=False, compression='snappy')
