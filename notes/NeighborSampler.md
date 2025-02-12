NeighborSampler
│
├── Initialization
│   ├── Takes adj_list (temporal graph structure)
│   ├── Stores neighbor IDs, edge IDs, and timestamps
│   └── Supports different sampling strategies (uniform/recent/time-aware)
│
├── Main Methods Used in Training
│   │
│   ├── find_neighbors_before(node_id, time)
│   │   └── Returns all interactions before given time
│   │
│   ├── get_historical_neighbors(node_ids, times, num_neighbors)
│   │   ├── Samples neighbors for each node
│   │   └── Returns (neighbor_ids, edge_ids, timestamps)
│   │
│   └── get_multi_hop_neighbors(num_hops, node_ids, times)
       └── Recursively samples neighbors for multiple hops

Flow in train_link_prediction.py:
┌─────────────────────┐
│ Initialize Samplers │
├─────────────────────┴──────────────────────┐
│ ├── train_neighbor_sampler                 │
│ │   └── Used during training               │
│ │                                          │
│ └── full_neighbor_sampler                  │
│     └── Used during validation/testing     │
│                                            │
├─────────────────────────────────────────────┤
│ Training Loop                               │
│ ├── For each batch:                        │
│ │   ├── Sample temporal neighbors          │
│ │   ├── Compute node embeddings            │
│ │   └── Update model                       │
│ │                                          │
│ └── For validation/testing:                │
│     └── Sample neighbors for evaluation    │
└─────────────────────────────────────────────┘


Time ─────────────────────────────────────────────────────────►
t=1    t=2    t=3    t=4    t=5    current_time=t=6
 │      │      │      │      │            │
 ▼      ▼      ▼      ▼      ▼            ▼
[U1]───[P1]   [U1]───[P4]   [U1]───[P7]   [U1]???[P?]
 │             │             │              
[U2]───[P2]   [U2]───[P5]   [U3]───[P8]    Want to predict:
 │             │             │              U1's next interaction
[U3]───[P3]   [U3]───[P6]   [U2]───[P9]