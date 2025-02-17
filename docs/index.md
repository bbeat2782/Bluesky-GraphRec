# Evaluating Graph Transformers for Scalable Social Media Recommendations

## Introduction

### What is the problem?
### Why is this important?

---

## Methods

### Overview
We use a **two-stage recommendation pipeline**:
1. **Candidate Generation**: Uses **collaborative filtering** and **consumer-producer embeddings** to efficiently generate a list of candidate posts for each user.
2. **Ranking**: Employs **GraphRec**, a transformer-based ranking model that incorporates:
   - Temporal graph learning
   - Node interaction embeddings
   - Efficient patching techniques for scalability

### Pipeline

<iframe src="assets/producer_embeddings.html" width="800" height="600" frameBorder="0"></iframe>

<div style="width: 100%; display: flex; justify-content: center;">
    <object type="image/svg+xml" data="assets/pipeline1.svg" style="width: 100%; max-height: 500px;"></object>
</div>

<div style="width: 100%; display: flex; justify-content: center;">
    <object type="image/svg+xml" data="assets/pipeline2.svg" style="width: 100%; max-height: 500px;"></object>
</div>



---

## Evaluation & Metrics

### Key Metrics
To evaluate our model, we focus on both **ranking quality** and **diversity**:
- **Hit Rate@K**: Measures how often the correct post appears in the top-K recommendations.
- **Recall@K**: Evaluates the model’s ability to retrieve relevant posts.
- **Mean Reciprocal Rank (MRR)**: Assesses ranking accuracy by considering the position of the first relevant recommendation.
- **Intra-List Diversity (ILD)**: Measures content variety within recommendations to avoid redundancy.

---

## Results
### Model Performance Comparison

### Performance Metrics

| Model        | MRR  | Average Rank | ILD |  Training Time (1 epoch) | Inference Time (sec/user) | 
|-------------|------|-----------|------|---|---|
| Popularity  | X.XX | X.XX      | X.XX | | |
| GraphMixer  | X.XX | X.XX      | X.XX | | |
| GraphRec | X.XX | X.XX | X.XX | | |

### Rank vs. Popularity Visualization

---

## Real-World Impact

---

## Future Work

---

## Acknowledgments

We thank our mentors and the UCSD DSC Capstone Program for their guidance and support.  

---

## References

[1] Yu, Le, Leilei Sun, Bowen Du, and Weifeng Lv. 2023.  
*"Towards Better Dynamic Graph Learning: New Architecture and Unified Library."*  
Available at: [arXiv:2303.13047](https://arxiv.org/abs/2303.13047).
