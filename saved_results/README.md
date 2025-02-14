# Generated Files

## During Training
For each run, the following files are generated:

- **Loss and Accuracy Plots**
  - `training_loss_plot_{seed}.png`, `training_accuracy_plot_{seed}.png`, `training_pairwise_accuracy_plot_{seed}.png`  
    - Visualizes training loss/accuracy/pairwise accuracy trends.

- **Training History**
  - `training_results_{seed}.json`  
    - Contains training histories for:
      - Training set
      - New node validation set
      - Validation set

- **Final Metrics**
  - `{modelname}_seed{seed}.json`  
    - After training is finished, stores accuracy and pairwise accuracy for:
      - Validation set
      - New node validation set
      - Test set
      - New node test set

## During Inference
After inference, the following files are generated:

- **Candidate Length Distribution**
  - `candidates_length_histogram.png`  
    - Histogram visualization of candidate lengths.
  - `candidates_length.json`  
    - Contains raw data used to generate the histogram.

- **MRR (Mean Reciprocal Rank) Results**
  - `mrr_results.npy`  
    - Stores MRR values computed for recommendations.
  - `real_negative_sampling_{modelname}_seed{seed}.json`  
    - Contains the final MRR score (average of all values in `mrr_results.npy`).

- **Recommended Posts**
  - `recommended_posts.json`  
    - Lists recommended posts in ranked order for all users.
