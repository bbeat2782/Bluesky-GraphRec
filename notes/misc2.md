#### Model Training
* Training *GraphRec* on *Bluesky* dataset:
```{bash}
python train_link_prediction.py --dataset_name bluesky --model_name GraphRec --patch_size 2 --max_input_sequence_length 64 --num_runs 1 --gpu 0 --batch_size 500 --negative_sample_strategy historical --num_epochs 30
```
#### Model Evaluation
* Evaluating *GraphRec* with posts that received at least one like in the last 20 minutes as candidate generation on *Bluesky* dataset:
```{bash}
python evaluate_link_prediction_v2.py --dataset_name bluesky --model_name GraphRec --patch_size 2 --max_input_sequence_length 64 --negative_sample_strategy real --num_runs 1 --gpu 0 --batch_size 4
```



Important files/locations:

models/GraphRec.py

utils/DataLoader.py

train_link_prediction.py

MRR: 
- evaluate_link_prediction_v2.py
- evaluate_models_utils.py


MRR: if the true item is ranked as first, then MRR is 1. If it's ranked second, then MRR is 0.5. If it's ranked third, then MRR is 0.33. And so on.


evaluation results:

list of reciprocal ranks, and also average, and also number of candidate items:

u1: 1
u2: 1/2
u3: 1/10

MRR: (1 + 1/2 + 1/10) / 3 = 0.63

number of candidate items: ~2000-4000


Candidate Generation Code:

utils/utils.py : CandidateEdgeSampler




Dataloader:

train: 1 million interactions

each sample: interaction between src_id and dst_id node, and the timestamp of the interaction, idx to get user dynamic feature

candidate generation: using src_id, dst_id, and timestamp to get everything before 20 minutes ago (posts that had 1 like at least 20 minutes ago)


Inputs to GraphRec Model:

1. node_raw_features:
   Shape: (num_total_nodes + 1, node_feat_dim) = (1000000, 128)
   Example:
   ```python
   node_raw_features = [
       [0, 0, ..., 0],          # Padding node (id=0)
       [0.1, 0.2, ..., 0.8],    # User 1's static features
       [0.2, 0.3, ..., 0.7],    # User 2's static features
       ...,
       [0.5, 0.6, ..., 0.9],    # Post 101's text embedding
       [0.4, 0.3, ..., 0.8],    # Post 102's text embedding
   ]
   ```

2. user_dynamic_features:
   Shape: (num_interactions, feature_dim) = (500000, 128)
   Example for User 1's interactions:
   ```python
   user_dynamic_features = [
       [0.0, 0.0, ..., 0.0],    # First interaction - no history
       [0.5, 0.6, ..., 0.9],    # Second interaction - avg of Post 101's embedding
       [0.45, 0.45, ..., 0.85], # Third interaction - avg of Posts 101 & 102
   ]
   ```

3. neighbor_sampler:
   Input: node_id=1, timestamp=1300
   Output example (for num_neighbors=2):
   ```python
   # Returns temporal neighborhood for User 1 before timestamp 1300
   neighbor_ids = [102, 101]       # Most recent neighbors first
   edge_ids = [2, 1]              # Corresponding edge IDs
   timestamps = [1200, 1000]      # When these interactions happened
   ```

4. time_feat_dim: int = 64
   Example time encoding for time difference Δt = current_time - neighbor_time:
   ```python
   # For Δt = 100 (time units):
   time_encoding = [
       cos(100 / 10^0),  # High frequency component
       cos(100 / 10^1),  # Medium frequency
       cos(100 / 10^2),  # Lower frequency
       ...,              # Captures different time scales
       cos(100 / 10^63)  # Very low frequency component
   ]
   ```

5. patch_size: int = 1
   Controls how neighbor sequences are grouped:
   ```python
   # If patch_size = 2:
   neighbors = [102, 101, 103, 104]  # 4 neighbors
   patches = [
       [102, 101],  # Patch 1
       [103, 104]   # Patch 2
   ]
   ```

6. max_seq_length: int = 512
   Maximum number of historical neighbors to consider:
   ```python
   # If user has 1000 historical interactions but max_seq_length = 512:
   actual_neighbors = neighbors[:512]  # Take most recent 512
   ```

Example of how they work together:


