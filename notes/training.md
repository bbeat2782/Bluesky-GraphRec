train_link_prediction.py

    get_link_prediction_data -> train_data, val_data, test_data, new_node_val_data, new_node_test_data

    train_neighbor_sampler ->

    train_neg_edge_sampler ->

    get all the data_loaders -> train_idx_data_loader, val_idx_data_loader, new_node_val_idx_data_loader, test_idx_data_loader, new_node_test_idx_data_loader

    GraphRec(all the parameters from above)

    model = nn.Sequential(GraphRec, LinkPredictor (this is a simple MLP))

    
    