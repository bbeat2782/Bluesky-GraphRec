[get_link_prediction_data] -> node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, user_dynamic_features


[get_neighbor_sampler] -> train_neighbor_sampler, full_neighbor_sampler


[MultipleNegativeEdgeSampler] -> train_neg_edge_sampler, val_neg_edge_sampler, new_node_val_neg_edge_sampler, test_neg_edge_sampler, new_node_test_neg_edge_sampler


[get_idx_data_loader] -> train_idx_data_loader, val_idx_data_loader, new_node_val_idx_data_loader, test_idx_data_loader, new_node_test_idx_data_loader


[GraphRec] -> dynamic_backbone, link_predictor


(train_data) -> batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids


[train_neg_edge_sampler] -> batch_neg_dst_node_ids

(np.repeat) x 4 -> batch_neg_src_node_ids, batch_neg_src_idx


[model.compute_src_dst_node_temporal_embeddings] -> batch_src_node_embeddings, batch_dst_node_embeddings


[model.compute_src_dst_node_temporal_embeddings] -> batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings


[flatten] -> don't understand this part

