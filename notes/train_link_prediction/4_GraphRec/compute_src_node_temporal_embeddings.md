get_all_first_hop_neighbors: user -> all post interactions

get_all_first_hop_neighbors: post -> all user interactions

padding:
# When sampling temporal neighbors, each node might have:
node1: [n1, n2, n3, n4, n5]        # 5 neighbors
node2: [n1, n2]                    # 2 neighbors
node3: [n1, n2, n3, n4, n5, n6]    # 6 neighbors

# We need equal length sequences for batch processing, so we pad:
node1: [n1, n2, n3, n4, n5, 0]     # padded to 6
node2: [n1, n2, 0, 0, 0, 0]        # padded to 6
node3: [n1, n2, n3, n4, n5, n6]    # already 6

truncated to max_input_sequence_length