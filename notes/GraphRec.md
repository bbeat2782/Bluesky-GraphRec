padding:
# When sampling temporal neighbors, each node might have:
node1: [n1, n2, n3, n4, n5]        # 5 neighbors
node2: [n1, n2]                    # 2 neighbors
node3: [n1, n2, n3, n4, n5, n6]    # 6 neighbors

# We need equal length sequences for batch processing, so we pad:
node1: [n1, n2, n3, n4, n5, 0]     # padded to 6
node2: [n1, n2, 0, 0, 0, 0]        # padded to 6
node3: [n1, n2, n3, n4, n5, n6]    # already 6

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