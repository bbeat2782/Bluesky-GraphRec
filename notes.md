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