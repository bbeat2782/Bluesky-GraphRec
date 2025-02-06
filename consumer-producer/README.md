```
--------------------------------------------
             DATA COLLECTION
--------------------------------------------
  [Training Data: Interactions before 06-15]
       |
       v
+-----------------------------+
| Build Bipartite Graph       |
| (Consumers <--> Producers)  |
+-----------------------------+
       |
       v
+-----------------------------+
| Matrix Factorization        |
| (SVD/NMF)                   |
| => Consumer Embeddings      |
+-----------------------------+
       |
       v
--------------------------------------------
             TEST PHASE
--------------------------------------------
[Test Data: Posts from 06-15 (or 06-15 to 06-16)]
       |
       v
+-----------------------------+
| For each Test Post:         |
|  - Collect Test Likes       |
|  - Compute Post Embeddings  |
|    using likes              |
+-----------------------------+
       |
       v
+-----------------------------+
| Build FAISS Index on Test   |
| Post Embeddings             | (FAISS is simply for efficient dot product/cosine similarity search)
+-----------------------------+
       |
       v
+-----------------------------+
| For each Test Consumer:     |
|  - Retrieve fixed consumer  |
|    embedding                |
|  - Query FAISS for Top-K    |
|    posts                    |
+-----------------------------+
       |
       v
+-----------------------------+
| Compare Recommendations vs. |
| Test Interactions           |
| (Compute Metrics)           |
+-----------------------------+
       |
       v
    Evaluation Results
--------------------------------------------
```

Notes:
- Producer affinity is leftover from previous work. We don't use it in the current implementation. You can safely ignore it.
- You can safely ignore clusters; for now we don't use them.

Offline evaluation limitations vs. Real-World realtime deployment:

1. In real-world, post embeddings are updated in realtime as new likes come in.
2. Offline evaluation is done on a fixed dataset of posts from 06-15 to 06-16. Post embeddings are computed on this fixed dataset ignoring time. Users may have liked something whose post embedding changes significantly after, although this seems rare.
3. User embeddings when recomputed nightly/hourly may have latent space change/drift. 

TODO:
- [x] Persistent post mappings
- [ ] GPU-accelerated FAISS index
- [ ] Persistent, deterministic consumer embeddings latent space
- [ ] Implement post embeddings for real-time
- [ ] like interactions -> post embeddings to 2nd model

