import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_recommendations(model, test_interactions, data, k_values=[20, 100, 1000], batch_size=128):
    """
    Evaluate the model using Recall@K for multiple K values with batched processing.
    
    Args:
        model: Trained LightGCN model
        test_interactions: Dict mapping user_ids to lists of their test post_ids
        data: PyG Data object containing the training graph
        k_values: List of K values to compute Recall@K for
        batch_size: Number of users to process at once
    """
    model.eval()
    recalls = {k: [] for k in k_values}
    
    # Convert test_interactions to lists for batching
    user_indices = list(test_interactions.keys())
    
    # Get all possible post indices (both from training and test sets)
    all_posts = set()
    for posts in test_interactions.values():
        all_posts.update(posts)
    post_indices = torch.tensor(list(all_posts)).to(device)
    
    with torch.no_grad():
        # Process users in batches
        for i in tqdm(range(0, len(user_indices), batch_size)):
            batch_users = user_indices[i:i + batch_size]
            
            # Prepare batch tensors
            src_index = torch.tensor(batch_users).to(device)
            
            # Adjust k_values if necessary
            max_possible_k = len(post_indices)
            actual_k_values = [min(k, max_possible_k) for k in k_values]
            
            # Get recommendations for the batch
            recommendations = model.recommend(
                edge_index=data.edge_index.to(device),
                src_index=src_index,
                dst_index=post_indices,
                k=min(max(k_values), max_possible_k)
            )
            
            # Process each user in the batch
            for user_idx, user_recs in zip(batch_users, recommendations):
                true_post_ids = test_interactions[user_idx]
                true_set = set(true_post_ids)
                
                # Calculate recall@k for each k
                for k, actual_k in zip(k_values, actual_k_values):
                    rec_set = set(user_recs[:actual_k].cpu().numpy())  # Move back to CPU for set operations
                    recall = len(rec_set.intersection(true_set)) / len(true_set) if true_set else 0
                    recalls[k].append(recall)
    
    # Calculate average recall for each K
    avg_recalls = {k: sum(scores)/len(scores) for k, scores in recalls.items()}
    
    return avg_recalls