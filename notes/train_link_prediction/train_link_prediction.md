train_link_prediction.py

[Imports & Setup]
- standard libraries (logging, time, sys, etc.)
- PyTorch
- Custom modules (GraphRec, MergeLayer, etc.)
- utils (get_neighbor_sampler, NegativeEdgeSampler, MultipleNegativeEdgeSampler, evaluate_model_link_prediction, get_link_prediction_metrics, get_idx_data_loader, get_link_prediction_data, EarlyStopping, get_link_prediction_args)

[get_link_prediction_data]
- Creates special test sets for "new nodes" (nodes not seen during training) to evaluate the model's inductive learning capabilities
- The key purpose is to prepare data for both transductive learning (predicting on known nodes) and inductive learning (predicting on unseen nodes).
- takes in graph_df, edge_raw_features, node_raw_features, and dynamic_user_features
- preprocesses the data to ensure compatibility with the model
- splits the data into train, val, and test sets
- returns node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, user_dynamic_features



[get_neighbor_sampler]
- train_neighbor_sampler, full_neighbor_sampler
- creates an adjacency list from the training data
- returns a NeighborSampler object
- [NeighborSampler]
    - get_unique_edges_between_start_end_time
    - sample
    - random_sample
    - historical_sample
    - inductive_sample

[MultipleNegativeEdgeSampler]
- samples multiple negative edges for each positive edge
- reshapes the output to have shape (batch_size, num_negatives) instead of just (batch_size,)
- looks at edges from the past 20 minutes
- used for efficient batch processing of multiple negative samples per positive edge
- train_neg_edge_sampler, val_neg_edge_sampler, new_node_val_neg_edge_sampler, test_neg_edge_sampler, new_node_test_neg_edge_sampler -> MultipleNegativeEdgeSampler objects


[get_idx_data_loader]
- creates a DataLoader object for the training, validation, and test sets
- returns a DataLoader object
- train_idx_data_loader, val_idx_data_loader, new_node_val_idx_data_loader, test_idx_data_loader, new_node_test_idx_data_loader -> DataLoader objects
    === Example data from dataloaders ===

    Training data examples:
    Number of training interactions: 10427384

    Example 1:
    Source node ID: 24219
    Destination node ID: 2454231
    Interaction time: 1672545122.0
    Edge ID: 3
    Label: 0
    Index: 3
    Total number of interactions: 10427384 (unique likes)
    Number of unique nodes: 3491300  (users and posts)
    Maximum source node ID: 106367

    Batch size: 512
    Number of batches in training: 20366


[Training Loop]
For each epoch:
    For each batch of 512 interactions:

    1. Get Batch Data:
       ```python
       # Each batch contains:
       batch_src_node_ids     # 512 users
       batch_dst_node_ids     # 512 posts they interacted with
       batch_node_interact_times  # When these interactions happened
       ```

    2. Sample Negative Edges:
       ```python
       # Repeat the source node ids 4 times match the number of negative samples
       batch_neg_dst_node_ids.shape = (512, 4)  # Each user gets 4 posts they didn't interact with
       ```

    3. Forward Pass:
       ```python
       # For positive edges
       positive_scores = model(users, posts_they_liked)
       
       # For negative edges
       negative_scores = model(users, posts_they_didnt_like)
       ```

    4. Compute Scores:
        ```python
        # For positive pairs
        positive_scores = model[1](
            input_1=batch_src_node_embeddings, 
            input_2=batch_dst_node_embeddings
        )  # Shape: (512,)

        # For negative pairs
        negative_scores = model[1](
            input_1=batch_neg_src_node_embeddings_flat,
            input_2=batch_neg_dst_node_embeddings_flat
        ).view(positive_scores.shape[0], 4)  # Shape: (512, 4)
        ```

    5. Compute Loss:
       ```python
       # Want: score(user->liked_post) > score(user->random_post)
       loss = -torch.log(
           torch.sigmoid(positive_scores - negative_scores)
       )
       ```

    6. Backward Pass:
       (backprops through the entire model (both parts))
       ```python
       loss.backward()
       optimizer.step()
       optimizer.zero_grad()
       ```

    Every 10 epochs:
      - Evaluate on validation set
      - Save model if validation improves
      - Early stop if no improvement for 5 evaluations

Key Metrics:
- MRR (Mean Reciprocal Rank): How well model ranks the true post vs random posts
- Training Loss: How well model distinguishes posts user liked vs didn't like


