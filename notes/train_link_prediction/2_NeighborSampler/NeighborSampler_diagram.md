NeighborSampler
│
├── __init__(adj_list)
│   │                         ┌─ nodes_neighbor_ids[0] = []
│   │   Input adj_list[0]=[] │  nodes_edge_ids[0] = []
│   └── Creates 4 lists ──────┤  nodes_neighbor_times[0] = []
│                             └─ nodes_neighbor_idx[0] = []
│
├── find_neighbors_before(node_id=1, time=1672542500)
│   │
│   │   Timeline for node 1:
│   │   1672541000    1672542000    1672542500
│   │        │             │             │
│   │        v             v             v
│   │   [neighbor=3]  [neighbor=4]    (query time)
│   │
│   └── Returns: ([3,4], [101,102], [1672541000,1672542000], None, [1,2])
│       neighbor_ids, edge_ids, times, None, neighbor_idx
│
│    multiple version of find_neighbors_before
├── get_historical_neighbors(node_ids=[1,2], times=[1672542500,1672543500], num_neighbors=2)
│   │
│   │                      ┌── For node 1 ──┐
│   │   Returns matrices:  │ [3 4]          │ neighbor_ids
│   └── Shape: (2,2) ─────┤ [101 102]      │ edge_ids
│                         │ [1672541000...] │ times
│                         └── For node 2 ──┘
│
└── get_multi_hop_neighbors(num_hops=2, node_ids=[1], times=[1672542500], num_neighbors=2)
    │
    │   Hop 1:           Hop 2:
    │   node 1           node 3    node 4
    │      │               │         │
    │      ├───────┐   ┌──┘         └──┐
    │      │       │   │               │
    │      v       v   v               v
    │   node 3  node 4  [neighbors]  [neighbors]
    │
    └── Returns: ([hop1_neighbor_ids, hop2_neighbor_ids], [hop1_edge_ids, hop2_edge_ids], [hop1_times, hop2_times])

Example inputs/outputs:

def __init__(adj_list=[[], [(3,101,1672541000,1), (4,102,1672542000,2)]]):
    """Creates data structures shown at top"""

def find_neighbors_before(node_id=1, time=1672542500):
    """
    Returns:
    - neighbor_ids: [3, 4]
    - edge_ids: [101, 102] 
    - times: [1672541000, 1672542000]
    - None
    - neighbor_idx: [1, 2]
    """

def get_historical_neighbors(node_ids=[1,2], times=[1672542500,1672543500], num_neighbors=2):
    """
    Returns:
    - neighbor_ids: [[3,4], [5,0]]  # 0 padded when <2 neighbors
    - edge_ids: [[101,102], [103,0]]
    - times: [[1672541000,1672542000], [1672543000,0]]
    """

def get_multi_hop_neighbors(num_hops=2, node_ids=[1], times=[1672542500], num_neighbors=2):
    """
    Returns:
    - hop1_neighbor_ids: [[3,4]]  # neighbors of node 1
    - hop2_neighbor_ids: [[1,0], [1,0]]  # neighbors of nodes 3,4
    - hop1_edge_ids: [[101,102]]
    - hop2_edge_ids: [[101,0], [102,0]]
    - hop1_times: [[1672541000,1672542000]]
    - hop2_times: [[1672541000,0], [1672542000,0]]
    """