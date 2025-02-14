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
            shape: (num_interactions, num_features) ex: (500000, 128)
            # Each row represents the rolling average of up to 10 previous post embeddings
            # for that user at the time of that interaction
            example:
            [
                [0.0, 0.0, ..., 0.0],         # First interaction for user1 (128 zeros)
                [0.2, 0.1, ..., 0.3],         # Second interaction - avg of previous post
                [0.3, 0.2, ..., 0.4],         # Third interaction - avg of prev 2 posts
                [0.0, 0.0, ..., 0.0],         # First interaction for user2
                [0.5, 0.3, ..., 0.2],         # Second interaction for user2
                ...`
            ]

        neighbor_sampler:
            shape: (num_interactions, num_neighbors) ex: (500000, 10)
            # Each row represents the 10 nearest neighbors of the user at the time of that interaction
            example:
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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