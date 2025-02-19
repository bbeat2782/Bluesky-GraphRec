change factorize.py to use the updated one. in consumer-producer folder.


find a solution to the latent space problem.

potential solutions:

1. Fixed Basis:
# First day determines basis
U, S, Vt = svd(initial_matrix)
# Subsequent days project using original basis
new_embeddings = new_data @ Vt.T[:k]

2. Procrustes Alignment:
# Align new embeddings to previous day's space
R = orthogonal_procrustes(current_embeds, previous_embeds)
aligned_embeds = current_embeds @ R

3. Incremental SVD:
# Update existing decomposition with new edges
updated_U, updated_S = update_svd(existing_U, existing_S, new_edges)

4. Anchor Users:
# Maintain consistent subspace using reference users
anchor_mask = [is_anchor_user(u) for u in all_users]
anchor_embeds = embeds[anchor_mask]
rotation = procrustes(previous_anchors, anchor_embeds)