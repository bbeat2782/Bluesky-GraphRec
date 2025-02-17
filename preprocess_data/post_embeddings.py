# TODO need to optimizer later + parametrize dataset path
# LATER search if there is other more efficient language models

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
# sample.csv should have two columns: `item_id` and `text`
unique_posts = pd.read_csv('../DG_data/bluesky/sample.csv')

# https://huggingface.co/jinaai/jina-embeddings-v3
# https://emschwartz.me/binary-vector-embeddings-are-so-cool/
# embedding into 128 dimension + binary quantization
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

# DataLoader for batching
batch_size = 1024*8
def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        if i % (100*500) == 0:
            print(i)
        yield data[i:i + batch_size]

# Process posts in batches
all_embeddings = []
for batch_texts in batch_generator(unique_posts['text'].tolist(), batch_size):
    embeddings = model.encode(batch_texts, truncate_dim=128)
    binary_embeddings = (embeddings > 0).astype(np.uint8)
    all_embeddings.extend(binary_embeddings)

# Convert embeddings to strings for CSV storage
unique_posts = unique_posts.drop(columns=['text'])
unique_posts['embeddings'] = [
    ','.join(map(str, emb)) for emb in all_embeddings
]

# Save DataFrame to CSV
output_csv_path = '../DG_data/bluesky/sample_with_embeddings.csv'
unique_posts.to_csv(output_csv_path, index=False)
# Check saved file
print(f"DataFrame saved to {output_csv_path}")
