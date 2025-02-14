class GraphRec(nn.Module):
    def __init__(...):
        """
        Key components:
        1. node_raw_features: User/post embeddings (128 dim)
        2. user_dynamic_tensor: Daily user states (64 dim)
        3. transformers: Process temporal interactions
        4. projection_layer: Maps features to common space
        """

    def compute_src_dst_node_temporal_embeddings(src_node_ids, dst_node_ids, times):
        """
        Main forward function:
        1. Get temporal neighbors for both users & posts
        2. Get features (node embeddings + time encodings)
        3. Process through transformers
        4. Return user & post embeddings
        """

    def compute_src_node_raw_features(src_node_ids, times):
        """
        Get user features:
        1. Get user's past interactions
        2. Pad sequences to same length
        3. Get user's dynamic state for that day
        4. Return processed features
        """

    def pad_sequences(...):
        """
        Make sequences same length:
        1. Find max sequence length
        2. Pad shorter sequences with zeros
        3. First position: current node
        4. Rest: temporal neighbors
        """

    def get_features(...):
        """
        Get node features:
        1. For users: Get dynamic state for that day
        2. For posts: Get static embeddings
        3. Add time encodings
        4. Return combined features
        """