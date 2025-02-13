import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from models.modules import TimeEncoder
from utils.utils import NeighborSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time

class GraphRecMultiCo(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu', max_user_feature_dim=2, user_dynamic_features=None, src_max_id=None):
        """
        GraphRec model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param channel_embedding_dim: int, dimension of each channel embedding
        :param patch_size: int, patch size
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        :param max_input_sequence_length: int, maximal length of the input sequence for each node
        :param device: str, device
        """
        super(GraphRecMultiCo, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float16)).to(device)

        # Extract all unique dates and users
        all_dates = sorted(user_dynamic_features.keys())  # Unique dates
        all_users = sorted({uid for date in user_dynamic_features for uid in user_dynamic_features[date]})  # Unique users
        
        num_dates = len(all_dates)
        num_users = len(all_users)
        embedding_dim = 64
        
        # Create index mappings
        date_to_index = {date: idx for idx, date in enumerate(all_dates)}
        user_to_index = {user_id: idx for idx, user_id in enumerate(all_users)}
        
        # Preallocate tensor (num_dates, num_users, embedding_dim)
        user_dynamic_tensor = torch.zeros((num_dates, num_users, embedding_dim), dtype=torch.float16, device="cuda")
        
        # Populate the tensor
        for date, users in user_dynamic_features.items():
            date_idx = date_to_index[date]  # Convert date to index
            for user_id, embedding in users.items():
                if user_id in user_to_index:  # Ensure user is mapped
                    user_idx = user_to_index[user_id]
                    user_dynamic_tensor[date_idx, user_idx] = torch.tensor(embedding, dtype=torch.float16, device="cuda")
        
        # Store lookup structures
        self.user_dynamic_tensor = user_dynamic_tensor
        self.date_to_index = date_to_index
        self.user_to_index = user_to_index

        self.device = device

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.src_max_id = src_max_id
        
        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        self.neighbor_co_occurrence_feat_dim = self.channel_embedding_dim
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(neighbor_co_occurrence_feat_dim=self.neighbor_co_occurrence_feat_dim, device=self.device)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.patch_size * self.neighbor_co_occurrence_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 3

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim, out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, batch_src_idx=None, is_eval=False):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param batch_src_idx: TODO for src nodes only (for ndynamic features)
        :return:
        """
        # get the first-hop neighbors of source and destination nodes
        # three lists to store source nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        # src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
        #     self.neighbor_sampler.get_multi_hop_neighbors(num_hops=self.walk_length, node_ids=src_node_ids,
        #                                                   node_interact_times=node_interact_times, num_neighbors=num_neighbors)
        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_multi_hop_neighbors(num_hops=2, node_ids=src_node_ids,
                                                          node_interact_times=node_interact_times, num_neighbors=8)

        # three lists to store destination nodes' first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        # dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
        #     self.neighbor_sampler.get_all_first_hop_neighbors(num_hops=self.walk_length, node_ids=dst_node_ids, node_interact_times=node_interact_times, num_neighbors=num_neighbors)
        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_multi_hop_neighbors(num_hops=2, node_ids=dst_node_ids, node_interact_times=node_interact_times, num_neighbors=8)

        batch_size = len(src_nodes_neighbor_ids_list[0])
        num_hops = 72  # (8 first-hop + 64 second-hop)
        
        # Preallocate arrays for source nodes
        src_padded_nodes_neighbor_ids = np.zeros((batch_size, num_hops), dtype=np.int32)
        src_padded_nodes_edge_ids = np.zeros((batch_size, num_hops), dtype=np.int32)
        src_padded_nodes_neighbor_times = np.zeros((batch_size, num_hops), dtype=np.float32)
        
        # Stack first-hop and second-hop neighbors efficiently
        src_padded_nodes_neighbor_ids[:, :8] = np.stack(src_nodes_neighbor_ids_list[0])
        src_padded_nodes_neighbor_ids[:, 8:] = np.stack(src_nodes_neighbor_ids_list[1])
        
        src_padded_nodes_edge_ids[:, :8] = np.stack(src_nodes_edge_ids_list[0])
        src_padded_nodes_edge_ids[:, 8:] = np.stack(src_nodes_edge_ids_list[1])
        
        src_padded_nodes_neighbor_times[:, :8] = np.stack(src_nodes_neighbor_times_list[0])
        src_padded_nodes_neighbor_times[:, 8:] = np.stack(src_nodes_neighbor_times_list[1])
        
        # Repeat for destination nodes
        dst_padded_nodes_neighbor_ids = np.zeros((batch_size, num_hops), dtype=np.int32)
        dst_padded_nodes_edge_ids = np.zeros((batch_size, num_hops), dtype=np.int32)
        dst_padded_nodes_neighbor_times = np.zeros((batch_size, num_hops), dtype=np.float32)
        
        dst_padded_nodes_neighbor_ids[:, :8] = np.stack(dst_nodes_neighbor_ids_list[0])
        dst_padded_nodes_neighbor_ids[:, 8:] = np.stack(dst_nodes_neighbor_ids_list[1])
        
        dst_padded_nodes_edge_ids[:, :8] = np.stack(dst_nodes_edge_ids_list[0])
        dst_padded_nodes_edge_ids[:, 8:] = np.stack(dst_nodes_edge_ids_list[1])
        
        dst_padded_nodes_neighbor_times[:, :8] = np.stack(dst_nodes_neighbor_times_list[0])
        dst_padded_nodes_neighbor_times[:, 8:] = np.stack(dst_nodes_neighbor_times_list[1])
        
        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # get the features of the sequence of source and destination nodes
        # src_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_max_seq_length, node_feat_dim)
        # src_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, src_max_seq_length, time_feat_dim)
        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=src_padded_nodes_edge_ids, padded_nodes_neighbor_times=src_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # dst_padded_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_max_seq_length, node_feat_dim)
        # dst_padded_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_max_seq_length, time_feat_dim)
        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                              padded_nodes_edge_ids=dst_padded_nodes_edge_ids, padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times, time_encoder=self.time_encoder)

        # get the patches for source and destination nodes
        # src_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, src_num_patches, patch_size * node_feat_dim)
        # src_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, src_num_patches, patch_size * time_feat_dim)
        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=src_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_neighbor_time_features=src_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        # dst_patches_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, dst_num_patches, patch_size * node_feat_dim)
        # dst_patches_nodes_neighbor_time_features, Tensor, shape (batch_size, dst_num_patches, patch_size * time_feat_dim)
        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(padded_nodes_neighbor_node_raw_features=dst_padded_nodes_neighbor_node_raw_features,
                             padded_nodes_neighbor_time_features=dst_padded_nodes_neighbor_time_features,
                             padded_nodes_neighbor_co_occurrence_features=dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)
        
        # align the patch encoding dimension
        # Tensor, shape (batch_size, src_num_patches, channel_embedding_dim)
        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_patches_nodes_neighbor_co_occurrence_features)

        # Tensor, shape (batch_size, dst_num_patches, channel_embedding_dim)
        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_patches_nodes_neighbor_co_occurrence_features)
      
        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, channel_embedding_dim)
        patches_nodes_neighbor_node_raw_features = torch.cat([src_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_neighbor_node_raw_features], dim=1)
        patches_nodes_neighbor_time_features = torch.cat([src_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_time_features], dim=1)
        patches_nodes_neighbor_co_occurrence_features = torch.cat([src_patches_nodes_neighbor_co_occurrence_features, dst_patches_nodes_neighbor_co_occurrence_features], dim=1)

        patches_data = [patches_nodes_neighbor_node_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features]
        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels, channel_embedding_dim)
        patches_data = torch.stack(patches_data, dim=2)

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        patches_data = patches_data.reshape(batch_size, src_num_patches + dst_num_patches, self.num_channels * self.channel_embedding_dim)

        # Tensor, shape (batch_size, src_num_patches + dst_num_patches, num_channels * channel_embedding_dim)
        for transformer in self.transformers:
            patches_data = transformer(patches_data)

        # src_patches_data, Tensor, shape (batch_size, src_num_patches, num_channels * channel_embedding_dim)
        src_patches_data = patches_data[:, : src_num_patches, :]
        # dst_patches_data, Tensor, shape (batch_size, dst_num_patches, num_channels * channel_embedding_dim)
        dst_patches_data = patches_data[:, src_num_patches: src_num_patches + dst_num_patches, :]
        # src_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        src_patches_data = torch.mean(src_patches_data, dim=1)
        # dst_patches_data, Tensor, shape (batch_size, num_channels * channel_embedding_dim)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_patches_data)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings


    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256, nodes_neighbor_idx_list=None, batch_src_idx=None):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
                # nodes_neighbor_idx_list[idx] = nodes_neighbor_idx_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)
        # padded_nodes_neighbor_idx = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]
            # padded_nodes_neighbor_idx[idx, 0] = batch_src_idx[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]
                # padded_nodes_neighbor_idx[idx, 1:len(nodes_neighbor_idx_list[idx]) + 1] = nodes_neighbor_idx_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times# , padded_nodes_neighbor_idx

    def get_user_embedding(self, date, user_id):
        if date in self.date_to_index and user_id in self.user_to_index:
            date_idx = self.date_to_index[date]
            user_idx = self.user_to_index[user_id]
            return self.user_dynamic_tensor[date_idx, user_idx]  # Fast GPU lookup
        else:
            return torch.zeros(64, dtype=torch.float16, device="cuda")  # Return zero vector if not found

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_idx: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        # Convert NumPy arrays to PyTorch tensors (on GPU)
        padded_nodes_neighbor_ids = torch.tensor(padded_nodes_neighbor_ids, dtype=torch.int64, device=self.device)
        padded_nodes_neighbor_times = torch.tensor(padded_nodes_neighbor_times, dtype=torch.int64, device=self.device)
    
        # Boolean mask for valid user nodes
        mask = padded_nodes_neighbor_ids <= self.src_max_id  # Shape: (batch_size, max_seq_length)

        # Retrieve raw features for all nodes
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[padded_nodes_neighbor_ids]
    
        ### **🔹 Fast User Feature Replacement (Fully Vectorized)**
        valid_indices = mask.nonzero(as_tuple=True)  # (batch_indices, seq_indices)
    
        if valid_indices[0].numel() > 0:  # Ensure we have valid nodes to process
            # Retrieve corresponding dates and user IDs
            interact_dates = padded_nodes_neighbor_times[valid_indices]  # Shape: (valid_count,)
            user_ids = padded_nodes_neighbor_ids[valid_indices]  # Shape: (valid_count,)
    
            ### **🔹 Convert date & user mappings into tensors for fast indexing**
            date_tensor = torch.tensor(list(self.date_to_index.keys()), dtype=torch.int64, device=self.device)
            date_map = torch.tensor(list(self.date_to_index.values()), dtype=torch.int64, device=self.device)
            user_tensor = torch.tensor(list(self.user_to_index.keys()), dtype=torch.int64, device=self.device)
            user_map = torch.tensor(list(self.user_to_index.values()), dtype=torch.int64, device=self.device)
    
            # Use torch.searchsorted for fast lookup
            date_indices = torch.searchsorted(date_tensor, interact_dates)
            user_indices = torch.searchsorted(user_tensor, user_ids)

            date_indices = torch.clamp(date_indices, min=0, max=date_map.shape[0] - 1)
            user_indices = torch.clamp(user_indices, min=0, max=user_map.shape[0] - 1)
    
            # Map indices using precomputed tensor (O(1) indexing)
            date_indices = date_map[date_indices]
            user_indices = user_map[user_indices]
    
            # Mask out invalid indices (date or user not found)
            valid_mask = (date_indices >= 0) & (user_indices >= 0)

            date_indices = date_indices[valid_mask]
            user_indices = user_indices[valid_mask]

            # Ensure date_indices and user_indices are the correct shape
            date_indices = date_indices.unsqueeze(1)  # Shape: (valid_count, 1)
            user_indices = user_indices.unsqueeze(1)  # Shape: (valid_count, 1)

            # Gather embeddings correctly
            user_embeddings = self.user_dynamic_tensor[date_indices, user_indices]  # Shape: (valid_count, 1, 64)

            user_embeddings = user_embeddings.squeeze(1)  # Shape: (valid_count, 64))

            # Pad user embeddings to 128 dimensions
            user_embeddings_padded = torch.zeros((user_embeddings.shape[0], 128), dtype=torch.float16, device=self.device)
            user_embeddings_padded[:, :64] = user_embeddings  # Avoids extra tensor concatenation

            user_embeddings_padded = user_embeddings_padded.to(padded_nodes_neighbor_node_raw_features.dtype)

            # padded_nodes_neighbor_node_raw_features[valid_indices[0], valid_indices[1], :] = user_embeddings_padded
            padded_nodes_neighbor_node_raw_features.index_put_(
                (valid_indices[0], valid_indices[1]), user_embeddings_padded
            )
 
        return padded_nodes_neighbor_node_raw_features.float(), padded_nodes_neighbor_time_features


    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_neighbor_time_features: torch.Tensor,
                    padded_nodes_edge_raw_features: torch.Tensor = None, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_co_occurrence_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_neighbor_time_features = [], []
        patches_nodes_neighbor_co_occurrence_features = []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_co_occurrence_features.append(padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)
        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.neighbor_co_occurrence_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs: torch.Tensor):
        """
        encode the inputs by Transformer encoder
        :param inputs: Tensor, shape (batch_size, num_patches, self.attention_dim)
        :return:
        """
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # Tensor, shape (num_patches, batch_size, self.attention_dim)
        transposed_inputs = inputs.transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        transposed_inputs = self.norm_layers[0](transposed_inputs)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs, key=transposed_inputs, value=transposed_inputs)[0].transpose(0, 1)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = inputs + self.dropout(hidden_states)
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.gelu(self.linear_layers[0](self.norm_layers[1](outputs)))))
        # Tensor, shape (batch_size, num_patches, self.attention_dim)
        outputs = outputs + self.dropout(hidden_states)
        return outputs


class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, neighbor_co_occurrence_feat_dim: int, device: str = 'cpu'):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.device = device

        self.neighbor_co_occurrence_encode_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_co_occurrence_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_co_occurrence_feat_dim, out_features=self.neighbor_co_occurrence_feat_dim))

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        device = self.device
    
        # Convert to PyTorch tensors for GPU processing
        src_padded_nodes = torch.tensor(src_padded_nodes_neighbor_ids, device=device)
        dst_padded_nodes = torch.tensor(dst_padded_nodes_neighbor_ids, device=device)

        batch_size, src_max_seq_length = src_padded_nodes.shape
        _, dst_max_seq_length = dst_padded_nodes.shape
    
        ### Step 1: Compute Unique Counts Batch-Wise Without Flattening ###
        
        # Compute unique counts per batch for source nodes
        src_unique, src_counts = torch.unique(src_padded_nodes, return_counts=True, dim=1)
        current_unique_nodes = src_counts.shape[0]

        if current_unique_nodes < 72:
            # Create zero-padded tensors
            src_unique_padded = torch.zeros((src_unique.shape[0], 72), device=src_unique.device, dtype=src_unique.dtype)
            src_counts_padded = torch.zeros((72,), device=src_counts.device)
            # Copy existing values into the padded tensors
            src_unique_padded[:, :current_unique_nodes] = src_unique
            src_counts_padded[:current_unique_nodes] = src_counts
        else:
            src_unique_padded = src_unique
            src_counts_padded = src_counts
        src_unique = src_unique_padded
        dst_unique, dst_counts = torch.unique(dst_padded_nodes, return_counts=True, dim=1)

        src_counts_padded = src_counts_padded.float()
        dst_counts = dst_counts.float()

        src_counts = src_counts_padded.unsqueeze(1)  # Shape: (batch_size, unique_count_per_batch, 1)
        dst_counts = dst_counts.unsqueeze(1)
    
        match_mask_src_src = src_padded_nodes.unsqueeze(2) == src_unique.unsqueeze(1)  # (batch_size, src_max_seq_length, dst_unique_length)
        src_padded_counts_in_src = torch.sum(match_mask_src_src * src_counts, dim=2)

        match_mask_dst_dst = dst_padded_nodes.unsqueeze(2) == dst_unique.unsqueeze(1)  # (batch_size, src_max_seq_length, dst_unique_length)
        dst_padded_counts_in_dst = torch.sum(match_mask_dst_dst * dst_counts, dim=2)

        # Step 2: Check if src nodes exist in dst_unique (vectorized lookup)
        match_mask_src = src_padded_nodes.unsqueeze(2) == dst_unique.unsqueeze(1)  # (batch_size, src_max_seq_length, dst_unique_length)
        src_padded_counts_in_dst = torch.sum(match_mask_src * dst_counts, dim=2)
    
        match_mask_dst = dst_padded_nodes.unsqueeze(2) == src_unique.unsqueeze(1)  # (batch_size, dst_max_seq_length, src_unique_length)
        dst_padded_counts_in_src = torch.sum(match_mask_dst * src_counts, dim=2)

        src_padded_nodes_appearances = torch.stack([src_padded_counts_in_src, src_padded_counts_in_dst], dim=-1)
        dst_padded_nodes_appearances = torch.stack([dst_padded_counts_in_src, dst_padded_counts_in_dst], dim=-1)
    
        # Zero out padded nodes (ensuring no spurious counts)
        src_padded_nodes_appearances[src_padded_nodes == 0] = 0.0
        dst_padded_nodes_appearances[dst_padded_nodes == 0] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                                                                                                  dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids)

        # sum the neighbor co-occurrence features in the sequence of source and destination nodes
        # Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)
        # Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances.unsqueeze(dim=-1)).sum(dim=2)

        # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features

