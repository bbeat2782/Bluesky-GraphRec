extract_from_duckdb.py -> ml_bluesky.csv, ml_bluesky.npy, ml_bluesky_node.npy

-> preprocess_data.py -> ml_bluesky.csv, ml_bluesky.npy, ml_bluesky_node.npy


# Data Preprocessing Pipeline

## 1. extract_from_duckdb.py
Takes raw DuckDB data and produces initial processed files.

### Input (DuckDB):
```sql
-- Example records table entry
{
    "collection": "app.bsky.feed.like",
    "record": {
        "subject": {
            "uri": "at://user123/app.bsky.feed.post/post456"
        }
    },
    "repo": "user789",
    "createdAt": "2023-05-20T15:30:00Z"
}
```

### Outputs:

#### bluesky.csv:
```csv
source_node,destination_node,timestamp,edge_label
0,1234,1684592400000,0
1,5678,1684592460000,0
```
- source_node: mapped user ID
- destination_node: mapped post ID
- timestamp: Unix timestamp in milliseconds
- edge_label: interaction type (0 for likes)

#### bluesky_text_embeddings.parquet:
```
item_id | embeddings
1234    | [0.1, -0.3, 0.5, ...]  # 128-dim float16 array
5678    | [-0.2, 0.4, 0.1, ...]
```

## 2. preprocess_data.py
Takes the initial processed files and formats them for model training.

### Outputs:

#### ml_bluesky.csv:
```csv
source_node,destination_node,timestamp,edge_label
10000,0,1684592400000,0  # Users start from 10000
10001,1,1684592460000,0  # Posts start from 0
```
- Reindexed to ensure proper bipartite graph structure
- Users and posts have non-overlapping ID ranges

#### ml_bluesky.npy:
```python
# Edge features array shape: [num_edges, feature_dim]
[
    [0.1, 0.2, ...],  # Feature vector for edge 1
    [0.3, 0.4, ...],  # Feature vector for edge 2
]
```

#### ml_bluesky_node.npy:
```python
# Node features array shape: [num_nodes, feature_dim]
[
    [0.5, 0.6, ...],  # Feature vector for node 0
    [0.7, 0.8, ...],  # Feature vector for node 1
]
```

Note: The actual feature dimensions and values will depend on the model configuration and embedding sizes used.