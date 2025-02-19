┌──────────────────────┐          ┌──────────────────────┐
│  Your Candidate Gen   │          │ Partner's GraphRec   │
│ (consumer-producer/)  │          │ (models/GraphRec.py) │
├──────────────────────┤          ├──────────────────────┤
│ 1. Factorize user-post│          │ 1. Temporal neighbor │
│    matrix → embeddings│          │    sampling          │
│ 2. FAISS index for    │◄┐     ┌─►│ 2. Transformer-based │
│    fast lookup        │ │     │  │    sequence encoding │
│ 3. Real-time post emb │ │     │  │ 3. BPR loss training │
└──────────────────────┘ │     │  └──────────────────────┘
                         │     │
                         │     │
┌───────────────────────────────────────────────────────┐
│                 Integration Points                    │
├───────────────────────────────────────────────────────┤
│ A. During Evaluation: Replace negative sampling with  │
│    your candidate generation                           │
│    (modify evaluate_models_utils.py)                  │
│                                                        │
│ B. Feature Integration: Inject your user/post embeddings│
│    into GraphRec's input features                      │
│    (modify DataLoader.py)                              │
│                                                        │
│ C. Dynamic Updates: Ensure embeddings stay fresh       │
│    (modify preprocess_data.py)                         │
└───────────────────────────────────────────────────────┘


# Detailed Component Analysis

## Candidate Generation System

                   ┌──────────────┐
                   │ DuckDB Data  │
                   └──────┬───────┘
                          ▼
               ┌──────────────────────┐
               │ Matrix Factorization │
               │  - user embeddings   │
               │  - post embeddings   │
               └───────┬──────────────┘
                       │
                       ▼
┌───────────────────────────────────────────┐
│ FAISS Index                               │
│  - Prebuilt post embeddings index         │
│  - Real-time nearest neighbor search      │
└───────┬───────────────────────────────────┘
        │
        ▼
┌──────────────────────┐
│ Candidate Generation │
│ - For each user:     │
│   1. Get user emb    │
│   2. FAISS query     │
│   3. Top 100 posts   │
└──────────────────────┘


## GraphRec System

┌───────────────────────────┐
│ Temporal Graph Processing │
│ 1. For each interaction:  │
│   - Sample temporal neigh-│
│     bors (past 20 mins)   │
│ 2. Transformer encoding  │
│   of neighbor sequences   │
│ 3. BPR loss with negative │
│    sampling               │
└─────────────┬─────────────┘
              ▼
┌───────────────────────────┐
│ Evaluation Pipeline        │
│ 1. Fixed candidate set     │
│    (posts w/ recent likes) │
│ 2. MRR calculation         │
└───────────────────────────┘



### Phase 1: Embedding Space Unification
1. **Align Dimensionality**
   - Modify: `models/GraphRec.py` input layers
   ```python
   # Original: node_feat_dim = 128
   NEW_DIM = 256  # Your emb (128) + Partner's (128)
   self.feature_proj = nn.Linear(NEW_DIM, 128)  # Project to expected dim
   ```
   *Rationale*: Prevent dimension mismatch when concatenating embeddings

2. **Temporal Alignment**
   - Modify: `preprocess_data.py`
   ```python
   def update_embeddings():
       # Your updated code with fixed basis
       daily_embeds = compute_stable_embeddings(new_data, 
               base_vt=initial_vt_matrix)  # Your fixed SVD basis
       # Partner's temporal features
       partner_features = load_partner_temporal_features() 
       aligned_embeds = align_using_procrustes(daily_embeds, partner_features)
   ```
   *Rationale*: Maintain temporal consistency across both systems

### Phase 2: Candidate Generation Integration
3. **Replace Negative Sampling**
   - Modify: `evaluate_models_utils.py` ~line 150
   ```python
   # Original: random negative sampling
   # Replace with:
   from consumer_producer.factorize import get_faiss_candidates
   batch_neg_dst_node_ids = get_faiss_candidates(
       src_embeddings=current_user_embs,
       timestamp=batch_node_interact_times
   )
   ```
   *Rationale*: Use your FAISS candidates instead of random negatives

4. **Feature Concatenation**
   - Modify: `utils/DataLoader.py` feature assembly
   ```python
   # Original: node_raw_features[idx]
   # Change to:
   combined_feat = np.concatenate([
       node_raw_features[idx],
       your_user_embeddings[user_id],
       your_post_embeddings[post_id]
   ], axis=-1)
   ```
   *Rationale*: Inject your embeddings into GraphRec's feature pipeline

### Phase 3: Dynamic Updating
5. **Real-time Refresh**
   - Create: `consumer-producer/streaming_updater.py`
   ```python
   class EmbeddingUpdater:
       def __init__(self):
           self.faiss_index = load_pretrained_index()
           self.graphrec_model = load_frozen_graphrec()
       
       def update(self, new_interactions):
           # Your incremental update logic
           new_embeds = factorize_incremental(new_interactions)
           self.faiss_index.add(new_embeds)
           # Partner's model update
           self.graphrec_model.update_temporal_features(new_interactions) 
   ```
   *Rationale*: Keep both systems in sync with new data

### Phase 4: Evaluation Bridge
6. **Metric Alignment**
   - Modify: `evaluate_models_utils.py` metrics calculation
   ```python
   def get_link_prediction_metrics():
       # Your MRR calculation
       your_mrr = calculate_your_mrr(predicts, labels)
       # Partner's existing metrics
       partner_metrics = original_metric_calculation()
       return {**partner_metrics, "your_mrr": your_mrr}
   ```
   *Rationale*: Compare both systems using shared metrics

### Phase 5: Training Pipeline
7. **Joint Training**
   - Modify: `train_link_prediction.py`
   ```python
   # Add to training loop:
   for batch in dataloader:
       your_scores = your_model(batch.user_emb, batch.post_emb)
       partner_scores = graphrec_model(batch.node_features)
       combined_loss = alpha*your_loss + (1-alpha)*partner_loss
       combined_loss.backward()
   ```
   *Rationale*: Allow both systems to co-train without catastrophic forgetting

### Key Integration Rationale
1. **Dimension Matching**: Prevent feature mismatch errors
2. **Temporal Coherence**: Align time-sensitive embeddings
3. **Candidate Quality**: Leverage FAISS for realistic negatives
4. **Feature Fusion**: Combine collaborative (your) and temporal (partner) signals
5. **Update Synchronicity**: Maintain consistency during live updates
6. **Metric Parity**: Enable direct comparison
7. **Joint Optimization**: Prevent model drift during co-training