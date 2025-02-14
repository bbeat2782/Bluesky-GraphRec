NeighborSampler object:

adj_list = [
    [],  # node 0: no connections
    [(3, 101, 1672541000, 1), (4, 102, 1672542000, 2)],  # node 1
    [(5, 103, 1672543000, 3)],  # node 2
    [(1, 101, 1672541000, 1)],  # node 3
    [(1, 102, 1672542000, 2)]   # node 4
]

# self.nodes_neighbor_ids - who they're connected to
nodes_neighbor_ids = [
    [],                  # node 0: no neighbors
    [3, 4],             # node 1: connected to nodes 3 and 4
    [5],                # node 2: connected to node 5
    [1],                # node 3: connected to node 1
    [1]                 # node 4: connected to node 1
]
  
# self.nodes_edge_ids - unique IDs for each connection
nodes_edge_ids = [
    [],                  # node 0: no edges
    [101, 102],         # node 1: edge IDs for connections
    [103],              # node 2: edge ID for connection
    [101],              # node 3: edge ID for connection
    [102]               # node 4: edge ID for connection
]

# self.nodes_neighbor_times - when each connection happened
nodes_neighbor_times = [
    [],                          # node 0: no timestamps
    [1672541000, 1672542000],   # node 1: timestamps of connections
    [1672543000],               # node 2: timestamp of connection
    [1672541000],               # node 3: timestamp of connection
    [1672542000]                # node 4: timestamp of connection
]

# self.nodes_neighbor_idx - index for each connection
nodes_neighbor_idx = [
    [],                  # node 0: no indices
    [1, 2],             # node 1: indices for connections
    [3],                # node 2: index for connection
    [1],                # node 3: index for connection
    [2]                 # node 4: index for connection
]