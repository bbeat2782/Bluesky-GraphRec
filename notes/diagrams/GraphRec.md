```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                Training Process                              │
├───────────────────────────────────┬─────────────────────────────────────────┤
│             Inputs                 │                Components               │
├───────────────────────────────────┼─────────────────────────────────────────┤
│                                   │                                           │
│ 1. Raw Data:                      │  [DataLoader.py]                         │
│    - ml_bluesky.csv               │  ┌────────────────────────────┐          │
│    (src, dst, timestamp)          │  │ get_link_prediction_data() │          │
│                                   │  └──────┬─────────────────────┘          │
│ 2. Node Features:                 │         │                                │
│    - ml_bluesky_node.npy          │         ▼                                │
│    (static)                       │  ┌────────────────────────────┐          │
│                                   │  │ NeighborSampler            │          │
│ 3. Edge Features:                 │  │ - Temporal sampling        │          │
│    - ml_bluesky.npy               │  │ - Padding/truncation       │          │
│                                   │  └──────┬─────────────────────┘          │
│ 4. User Dynamics:                 │         │                                │
│    - user_dynamic_features.pkl    │         ▼                                │
│    (rolling averages)             │  ┌────────────────────────────┐          │
│                                   │  │ NegativeEdgeSampler        │          │
│                                   │  │ - 4 neg samples per pos    │          │
│                                   │  │ - Same src, random dst     │          │
│                                   │  └──────┬─────────────────────┘          │
│                                   │         │                                │
│                                   │         ▼                                │
│                                   │  ┌────────────────────────────┐          │
│                                   │  │ GraphRec Model              │          │
├───────────────────────────────────┼──┤ (models/GraphRec.py)        │          │
│          Processing Steps          │  │ 1. Neighbor retrieval      │          │
│                                   │  │ 2. Time encoding           │          │
│                                   │  │ 3. Patch creation          │          │
│                                   │  │ 4. Transformer layers      │          │
│                                   │  └──────┬─────────────────────┘          │
│                                   │         │                                │
│                                   │         ▼                                │
│                                   │  ┌────────────────────────────┐          │
│                                   │  │ MergeLayer (MLP)           │          │
│                                   │  │ - src + dst → score        │          │
│                                   │  └──────┬─────────────────────┘          │
│                                   │         │                                │
│                                   │         ▼                                │
│                                   │  ┌────────────────────────────┐          │
│                                   │  │ BPR Loss Calculation       │          │
│                                   │  │ - pos vs neg scores        │          │
│                                   │  └──────┬─────────────────────┘          │
│                                   │         │                                │
├───────────────────────────────────┼─────────┴───────────────────────────────┤
│             Outputs                │                Artifacts               │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ - Trained Model Weights           │  saved_models/GraphRec/bluesky/         │
│ - Training Metrics                 │  saved_results/.../training_results.json│
│ - Validation Metrics               │                                         │
└───────────────────────────────────┴─────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                             Inference/Prediction Process                     │
├───────────────────────────────────┬─────────────────────────────────────────┤
│             Inputs                 │                Components               │
├───────────────────────────────────┼─────────────────────────────────────────┤
│                                   │                                           │
│ 1. New Interaction:              │  [evaluate_models_utils.py]              │
│    - src_node_id                  │  ┌────────────────────────────┐          │
│    - dst_node_id                  │  │ CandidateEdgeSampler       │          │
│    - timestamp                    │  │ - 100 neg candidates       │          │
│                                   │  │ - Same src, random dst     │          │
│ 2. Model Weights:                │  └──────┬─────────────────────┘          │
│    - GraphRec_seed2025.pkl       │         │                                │
│                                   │         ▼                                │
│ 3. Historical Data:               │  ┌────────────────────────────┐          │
│    - Past interactions           │  │ compute_temporal_embeddings │          │
│    - Existing embeddings         │  │ - Neighbor aggregation      │          │
│                                   │  │ - Transformer processing    │          │
│                                   │  └──────┬─────────────────────┘          │
│                                   │         │                                │
├───────────────────────────────────┼─────────┴───────────────────────────────┤
│             Outputs                │                Results                  │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ - Prediction Scores:              │  ┌────────────────────────────┐         │
│   [positive_score, neg1, neg2,...]│  │ Metrics:                   │         │
│                                   │  │ - MRR                      │         │
│ - Ranking Position:              │  │ - AUC                      │         │
│   (e.g. 1st, 5th, etc.)           │  │ - Recall@k                 │         │
│                                   │  └────────────────────────────┘         │
└───────────────────────────────────┴─────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              Key Data Transformations                        │
├───────────────────────────────────┬─────────────────────────────────────────┤
│             Stage                 │                Tensor Shapes            │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ Raw Input:                        │ (batch_size,)                            │
│ - Node IDs                        │                                          │
│                                   │                                          │
│ After Neighbor Sampling:          │ (batch_size, max_seq_length)             │
│ - Padded neighbor sequences       │                                          │
│                                   │                                          │
│ After Time Encoding:              │ (batch_size, max_seq_length, 64)        │
│ - Time delta features             │                                          │
│                                   │                                          │
│ After Patching:                   │ (batch_size, num_patches, patch_size*   │
│                                   │  (node_feat_dim + time_feat_dim))        │
│                                   │                                          │
│ Transformer Output:               │ (batch_size, embedding_dim)             │
│                                   │                                          │
│ Final Prediction:                │ (batch_size, 1)                         │
└───────────────────────────────────┴─────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              Model Architecture Details                     │
├───────────────────────────────────┬─────────────────────────────────────────┤
│ Component                         │ Function                                │
├───────────────────────────────────┼─────────────────────────────────────────┤
│ Time Encoder                      │ Convert Δt to cosine features           │
│                                   │ (64-dim)                                │
│                                   │                                         │
│ Neighbor Aggregator               │ Pad/truncate to 512 neighbors           │
│                                   │                                         │
│ Patch Creator                     │ Group into patches of 2-4 neighbors     │
│                                   │                                         │
│ Transformer Layers                │ 4-8 layers with multi-head attention    │
│                                   │                                         │
│ Projection Head                   │ Reduce to original feature dimensions   │
│                                   │                                         │
│ Merge Layer                       │ Concatenate src+dst → MLP → score       │
└───────────────────────────────────┴─────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                              GraphRec Model Architecture                     │
├───────────────────────────┬───────────────────┬─────────────────┬───────────┤
│ Component                  │ Sub-components     │ Parameters      │ Input/Output Shapes │
├───────────────────────────┼───────────────────┼─────────────────┼───────────┤
│ Input                     │                   │                 │           │
│ - src_node_ids            │                   │ batch_size      │ (B,)     │
│ - dst_node_ids            │                   │                 │           │
│ - node_interact_times     │                   │                 │           │
│ - edge_ids                │                   │                 │           │
│ - src_idx (dynamic feat)  │                   │                 │           │
├───────────────────────────┼───────────────────┼─────────────────┼───────────┤
│ Neighbor Sampler          │ 1. Temporal       │ num_neighbors=20│           │
│ (utils/DataLoader.py)     │    neighbor lookup│ max_seq_length=512│ (B, 512) │
│                           │ 2. Padding/       │                 │           │
│                           │    truncation     │                 │           │
├───────────────────────────┼───────────────────┼─────────────────┼───────────┤
│ Feature Assembly          │ 1. Static Features│ node_feat_dim=128│          │
│                           │   - node_raw_features│               │ (B, 512, 128) │
│                           │ 2. Dynamic Features│ dynamic_feat_dim=64│       │
│                           │   - user_dynamic_features│          │ (B, 512, 64) │
│                           │ 3. Time Encoder    │ time_feat_dim=64│           │
│                           │   Δt = t_cur - t_hist│              │ (B, 512, 64) │
├───────────────────────────┼───────────────────┼─────────────────┼───────────┤
│ Patch Creation            │ 1. Group neighbors │ patch_size=2   │           │
│ (models/GraphRec.py)      │ 2. Concatenate     │                 │ (B, 256, 256) │
│                           │    [node_feat ‖ Δt_feat] │          │           │
│                           │ 3. Linear projection│               │ (B, 256, 128) │
├───────────────────────────┼───────────────────┼─────────────────┼───────────┤
│ Transformer Encoder       │ 1. Multi-head     │ num_heads=8    │           │
│ (4 layers)                │    Attention      │ dim_feedforward=512│       │
│                           │ 2. Layer Norm      │ dropout=0.1    │ (B, 256, 128) │
│                           │ 3. FFN             │                 │           │
├───────────────────────────┼───────────────────┼─────────────────┼───────────┤
│ Output Pooling            │ 1. Mean pooling    │                 │           │
│                           │    across patches  │                 │ (B, 128) │
│                           │ 2. Linear projection│               │           │
├───────────────────────────┼───────────────────┼─────────────────┼───────────┤
│ Merge Layer (MLP)         │ 1. Concatenate     │ hidden_dim=256  │           │
│ (models/modules.py)       │    src + dst embeds│ num_layers=2   │ (B, 1)   │
│                           │ 2. FC layers       │                 │           │
├───────────────────────────┼───────────────────┼─────────────────┼───────────┤
│ BPR Loss Calculation      │ 1. Positive scores│ num_negatives=4 │           │
│                           │ 2. Negative scores│ margin=1.0      │           │
│                           │ 3. Log-sigmoid    │                 │           │
└───────────────────────────┴───────────────────┴─────────────────┴───────────┘

Data Flow:
(B = batch_size, typically 512)

Node IDs → [Neighbor Sampler] → Padded Neighbor Sequences → 
[Feature Assembly] → 
(Node Features ‖ Dynamic Features ‖ Time Features) → 
[Patch Creation] → 
Patched Sequences → 
[Transformer Encoder] → 
Contextualized Embeddings → 
[Mean Pooling] → 
Node Embeddings → 
[Merge Layer] → 
Interaction Scores

Key Operations:
1. Time Encoding: 
   φ(Δt) = concat([cos(Δt/100^(i/63)) for i in range(64)])

2. Patch Projection:
   W_patch ∈ ℝ^(192×128)  # (128 node + 64 time) → 128-d

3. Transformer Attention:
   Attention(Q,K,V) = softmax(QK^T/√d)V

4. BPR Loss: 
   L = -log(σ(pos_score - neg_scores)).mean()
```