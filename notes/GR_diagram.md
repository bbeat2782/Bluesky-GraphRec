INPUT: User-Post Interaction
[user_123] ----interacts with----> [post_456] @ timestamp_t
    |                                  |
    v                                  v
    
STEP 1: Neighbor Sampling & Feature Collection
[user_123]                         [post_456]
    |                                  |
    v                                  v
Past Neighbors:                    Past Neighbors:  
u_1 @ t-5min                      p_1 @ t-3min
u_2 @ t-10min                     p_2 @ t-7min
u_3 @ t-15min                     p_3 @ t-12min
    |                                  |
    v                                  v

STEP 2: Create Patches (patch_size=1 by default)
[user patches]                     [post patches]
    |                                  |
    v                                  v
┌─────────────┐                  ┌─────────────┐
│ node_feat   │                  │ node_feat   │
│ time_feat   │                  │ time_feat   │
└─────────────┘                  └─────────────┘
      |                                |
      v                                v

STEP 3: Project Patches
┌─────────────┐                  ┌─────────────┐
│ Projection  │                  │ Projection  │
│   Layer     │                  │   Layer     │
└─────────────┘                  └─────────────┘
      |                                |
      v                                v

STEP 4: Transformer Layers
┌─────────────────────────────────────────────┐
│              Transformer Block              │
│   [Attention across all patches & channels] │
└─────────────────────────────────────────────┘
                     |
                     v
┌─────────────────────────────────────────────┐
│              Transformer Block              │
│             (num_layers times)              │
└─────────────────────────────────────────────┘
        |                          |
        v                          v

STEP 5: Mean Pooling
[user_embedding]              [post_embedding]
        |                          |
        v                          v

STEP 6: Link Prediction (MergeLayer)
┌─────────────────────────────────────────────┐
│     score = MLP([user_emb || post_emb])     │
└─────────────────────────────────────────────┘
                     |
                     v
OUTPUT: Interaction Score (0 to 1 after sigmoid)
                   [0.82]

During evaluation, for each (user, timestamp) pair:
┌────────────────────────────────────────┐
│ true_post: 0.82                        │
│ candidate_1: 0.45                      │
│ candidate_2: 0.31    => MRR = 1.0      │
│ candidate_3: 0.12    (ranked 1st)      │
└────────────────────────────────────────┘


class GraphRec(nn.Module):
    def forward(self, src_node_ids, dst_node_ids, node_interact_times, batch_src_idx=None):
        """Main entry point for training
        Computes embeddings and returns interaction scores for source-destination pairs
        """

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times, batch_src_idx=None):
        """Computes embeddings for both source and destination nodes
        1. Gets source node embeddings
        2. Gets destination node embeddings 
        Returns both for scoring
        """

    def compute_src_node_temporal_embeddings(self, src_node_ids, node_interact_times, batch_src_idx=None):
        """Computes embeddings just for source nodes
        1. Samples temporal neighbors
        2. Gets features & creates patches
        3. Runs through transformer
        Returns source embeddings
        """

    def pad_sequences(self, node_ids, node_interact_times, nodes_neighbor_ids_list, ...):
        """Pads neighbor sequences to same length
        1. Cuts sequences longer than max_length
        2. Pads shorter sequences with zeros
        3. Ensures sequences are divisible by patch_size
        Returns padded arrays of neighbor IDs, edge IDs, times
        """

    def get_features(self, node_interact_times, padded_nodes_neighbor_ids, ...):
        """Gets raw features for nodes and time encoding
        1. Gets node features (static or dynamic)
        2. Computes time encoding between current time and neighbor times
        Returns node features and time features
        """

    def get_patches(self, padded_nodes_neighbor_node_raw_features, ...):
        """Groups node+time features into patches
        1. Splits sequences into patches of size patch_size
        2. Reshapes features within each patch
        Returns patches ready for transformer
        """

    def get_user_embedding(self, date, user_id):
        """Gets dynamic user features for a specific date
        Returns cached tensor if exists, else zeros
        """


forward()
    └─> compute_src_dst_node_temporal_embeddings()
            ├─> compute_src_node_temporal_embeddings()  # for source
            │       ├─> pad_sequences()  # pad neighbor lists
            │       ├─> get_features()   # get node & time features  
            │       └─> get_patches()    # create patches
            │
            └─> (same for destination nodes)
                    └─> MergeLayer scoring

evaluate_real()
    └─> compute_src_dst_node_temporal_embeddings()
            ├─> (process source node)
            └─> (process each candidate destination)
                    └─> rank candidates by scores

