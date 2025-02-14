get_link_prediction_data


input:
- graph_df
    - src_node_ids
    - dst_node_ids
    - edge_ids
    - node_interact_times
    - labels


- edge_raw_features
- node_raw_features
- dynamic_user_features_path



output:
- full_data
- train_data
- val_data
- test_data
- new_node_val_data
- new_node_test_data









