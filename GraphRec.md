Dataloader

src_node_ids = [1, 2, 3]
dst_node_ids = [101, 102, 103]
node_interaction_time = [1717334400, 1717334400, 1717334400]


bluesky.csv:

    source_node	destination_node	timestamp	edge_label
    0	12248	1349	20230101024321	0
    1	50947	3044497	20230101024954	0
    2	24218	2347863	20230101035202	0
    3	13743	1349	20230101051655	0
    4	50947	1349	20230101053502	0
    5	50947	3460233	20230101054209	0
    ...



DataLoader.py:
    get_link_prediction_data():
        node_raw_features:
            shape: (num_nodes, num_features) ex: (1000000, 128)
        user_dynamic_features:
            structure: Dict[timestamp, Dict[user_id, np.array]]
            # Organized by timestamp -> user -> feature vector
            # Loaded from: /home/sgan/private/DyGLib/DG_data/bluesky/user_dynamic_features.pkl
            # Originally computed in preprocess_data.py (currently commented out):
            # - Takes each user's interaction history
            # - For each interaction, averages previous post embeddings (up to 10)
            # - Groups by timestamp for efficient lookup
            example:
            {
                108: {  # The 108th unique day in the dataset
                    # Groups all users' states during this day
                    16780: [-0.311, 0.179, ...],  # User's avg post history
                    37902: [-0.036, 0.055, ...],  # Another user's history
                    100915: [0.0, 0.0, ...],      # New user (no history)
                },
                109: {  # The 109th day
                    # User states during the next day
                }
            }

        neighbor_sampler:
            # Example initial data:
            interactions = [
                # (src_id, dst_id, timestamp, edge_id)
                (1, 101, 1000, 1),  # User 1 interacts with Post 101
                (1, 102, 1200, 2),  # User 1 interacts with Post 102 
                (1, 103, 1400, 3),  # User 1 interacts with Post 103
                (2, 201, 1100, 4),  # User 2 interacts with Post 201
                (2, 202, 1300, 5)   # User 2 interacts with Post 202
            ]

            # The sampler first builds adjacency lists for each node:
            adj_list = [
                [],  # Empty list for node 0 (not used)
                [  # Neighbors for node 1 (User 1)
                    (101, 1, 1000, 1),  # (neighbor_id, edge_id, timestamp, idx)
                    (102, 2, 1200, 2),
                    (103, 3, 1400, 3)
                ],
                [  # Neighbors for node 2 (User 2) 
                    (201, 4, 1100, 4),
                    (202, 5, 1300, 5)
                ],
                # ... and so on for all nodes
            ]
        

+       edge_raw_features:
+           shape: (num_edges + 1, edge_feat_dim) = (500000, 2) 
+           # Currently placeholder zeros and not actively used in model.
+           # Edge information is only used for:
+           # 1. Indexing/tracking interactions in neighbor sampling
+           # 2. Maintaining the graph structure
+           # 3. Future extensibility - could be used to add interaction features
            # Note: While edge_raw_features are loaded in DataLoader.py,
            # only edge_ids are actually used in the Data class and 
            # subsequent processing. The features themselves are just
            # passed through without being used.
            example = [
                [0.0, 0.0],     # Index 0: Padding edge
                [0.0, 0.0],     # Edge 1: User 1 -> Post 101 (only used for tracking)
                [0.0, 0.0],     # Edge 2: User 1 -> Post 102
                ...
            ]


                                          ┌────────────────────────────┐
                                          │     Data Files             │
                                          │ ─ CSV (e.g., ml_bluesky.csv) │
                                          │ ─ Numpy (.npy, .npz) files  │
                                          └─────────────┬──────────────┘
                                                        │
                                                        │
                                                        ▼
                        ┌────────────────────────────────────────────┐
                        │      get_link_prediction_data()            │
                        │                                            │
                        │   Read CSV & Features:                     │
                        │     • src_node_ids, dst_node_ids            │
                        │     • timestamps (ts), labels               │
                        │     • node_raw_features, edge_raw_features   │
                        │     • user_dynamic_features (from NPZ)       │
                        │                                            │
                        │   Compute time bounds (e.g., val_time,       │
                        │   test_time via quantile calculations)       │
                        │                                            │
                        │   Split into:                              │
                        │     • full_data, train_data,                │
                        │       val_data, test_data, etc.              │
                        └─────────────┬──────────────────────────────┘
                                      │
                                      ▼
                   ┌─────────────────────────────────────┐
                   │          DataLoader                 │
                   │  (CustomizedDataset; batching of     │
                   │   indices via get_idx_data_loader)    │
                   └─────────────┬───────────────────────┘
                                 │
                                 ▼
                   ┌──────────────────────────────────────┐
                   │         NeighborSampler              │
                   │  (Prepares temporal neighbor info    │
                   │   for each node based on interaction)  │
                   └─────────────┬────────────────────────┘
                                 │
                                 ▼
         ┌────────────────────────────────────────────────────────────┐
         │                      GraphRec Model                        │
         │  (Main backbone for computing temporal node embeddings)    │
         │                                                            │
         │   Input: node_raw_features, user_dynamic_features,         │
         │          neighbor_sampler, time_feat_dim,                  │
         │          patch_size, max_seq_length, etc.                  │
         │                                                            │
         │   ┌────────────────────────────────────────────────────┐   │
         │   │        1. Neighbor Retrieval & Padding             │   │
         │   │   • For a batch of src & dst node IDs:             │   │
         │   │       - Get first-hop neighbor IDs, edge IDs,       │   │
         │   │         neighbor times (via sampler)                │   │
         │   │       - Pad sequences to a fixed length (with       │   │
         │   │         the target node prepended)                 │   │
         │   └────────────────────────────────────────────────────┘   │
         │                                                            │
         │   ┌────────────────────────────────────────────────────┐   │
         │   │        2. Feature Extraction & Patch Creation      │   │
         │   │   • Use raw features for each neighbor             │   │
         │   │   • Compute time difference and encode it with     │   │
         │   │     TimeEncoder (cosine transformation)            │   │
         │   │   • Segment padded sequences into patches          │   │
         │   │     (patch_size × feature_dim per patch)           │   │
         │   └────────────────────────────────────────────────────┘   │
         │                                                            │
         │   ┌────────────────────────────────────────────────────┐   │
         │   │    3. Projection & Transformer Encoding            │   │
         │   │   • Project each patch separately:                 │   │
         │   │       - Node patch via linear layer                │   │
         │   │       - Time patch via linear layer                │   │
         │   │   • Concatenate features (2 channels: node & time)   │   │
         │   │   • Process through several TransformerEncoder     │   │
         │   │     layers (self-attention, skip-connections)        │   │
         │   └────────────────────────────────────────────────────┘   │
         │                                                            │
         │   ┌────────────────────────────────────────────────────┐   │
         │   │           4. Output Embedding                      │   │
         │   │   • Mean-pool the patches' outputs                 │   │
         │   │   • Use an output linear layer to get final         │   │
         │   │     node embeddings (same dimensionality as raw)     │   │
         │   └────────────────────────────────────────────────────┘   │
         └─────────────┬────────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────────────────────┐
         │           MergeLayer (LinkPredictor)        │
         │  (Combines source and destination embeddings)│
         │  • Concatenation followed by FC layers         │
         │  • Outputs a score (for an edge)               │
         └─────────────┬───────────────────────────────┘
                       │
                       ▼
         ┌──────────────────────────────────────────────┐
         │         BPR Loss Computation &             │
         │          Negative Sampling                 │
         │  • For each positive edge (src → dst):       │
         │      - Sample negative replacements            │
         │      - Compare positive vs. negatives          │
         │      - Compute BPR loss (maximize score diff)    │
         └─────────────┬──────────────────────────────┘
                       │
                       ▼
         ┌──────────────────────────────────────────────┐
         │              Optimizer Step                 │
         │  (Back-propagation, weight update)          │
         └──────────────────────────────────────────────┘
                       │
                       ▼
         ┌──────────────────────────────────────────────┐
         │           Evaluation & Metrics              │
         │  • Running evaluation on:                   │
         │      - Validation set                        │
         │      - Test set (including inductive splits) │
         │  • Compute link prediction metrics            │
         └──────────────────────────────────────────────┘