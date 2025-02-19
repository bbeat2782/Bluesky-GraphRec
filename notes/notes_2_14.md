data split = 0.7, 0.15, 0.15\

new_node_val_data, new_node_test_data

padding: 
adj[0] = [] ; its the only one that's empty


subset is the last portion (10%) of the data

candidate generation part: utils.py/CandidateEdgeSampler
used in evaluation


sort_by_popularity and get_unique_posts_between_start_end_time use the same candidates.

for each pos sample, you get 4 neg samples.


 batch_neg_dst_node_ids.shape: (512, 4)
batch_neg_dst_node_ids: [[4145387  303234  550967 5043429]
 [ 626064 2179717  980280  106981]
 [5488843 4270608 3336469 1635801]
 [ 219748 2424624  832581  286156]
 [ 670506 2344843 1437760 5851298]]

flattened:

 [4145387  303234  550967 5043429 626064 2179717  980280  106981 5488843 4270608 3336469 1635801 219748 2424624  832581  286156 670506 2344843 1437760 5851298]



[go print out the stuff from compute_src_node_temporal_embeddings]


get_patches concatenates the features

GraphRec shapes in GraphRec.py


user_dynamic_features.pkl calculated for each day on the entire network/data.




input: src_node_ids, dst_node_ids, node_interact_times

GraphRec: 
    - patch
    - project
    - concat

    bpr loss
    backprop

MergeLayer(MLP):
    - predict

eval