import json
import numpy as np
import matplotlib.pyplot as plt

# Load JSON data
graphrec_path = 'GraphRecMultiCo/bluesky/training_results_42_1.json'  # Renamed for clarity
mlp_path = 'TGAT/bluesky/training_results_42_1.json'

with open(graphrec_path, 'r') as f:
    graphrec_data = json.load(f)

with open(mlp_path, 'r') as f:
    mlp_data = json.load(f)

# Updated metric names for the report
metrics = {
    "loss": "BPR Loss",
    "acc": "Accuracy",
    "pairwise_acc": "Pairwise Accuracy"
}

histories = {
    "loss": ["train_loss_history", "val_loss_history", "new_val_loss_history"],
    "acc": ["train_acc_history", "val_acc_history", "new_val_acc_history"],
    "pairwise_acc": ["train_pairwise_acc_history", "val_pairwise_acc_history", "new_val_pairwise_acc_history"]
}

# Updated line styles for better distinction
line_styles = {
    "train": "-",      # Solid line for training
    "val": "--",       # Dashed line for validation
    "new_val": ":"     # Dotted line for new validation
}

# Model colors
colors = {"GraphRec": "blue", "MLP-Based": "red"}

# Output file names
plot_filenames = {
    "loss": "combined/bpr_loss_comparison.png",
    "acc": "combined/accuracy_comparison.png",
    "pairwise_acc": "combined/pairwise_accuracy_comparison.png"
}

# Generate and save each metric plot
for i, (metric_key, metric_label) in enumerate(metrics.items()):
    plt.figure(figsize=(8, 5))

    for model_name, model_data in [("GraphRec", graphrec_data), ("MLP-Based", mlp_data)]:
        for history_key in histories[metric_key]:
            if history_key in model_data:
                label_name = history_key.replace("_history", "").replace("_", " ").title()
                linestyle = (
                    line_styles["train"] if "train" in history_key 
                    else line_styles["val"] if "val" in history_key and "new" not in history_key 
                    else line_styles["new_val"]
                )
                
                # Ensure epochs start from 1
                epochs = np.arange(1, len(model_data[history_key]) + 1)
                plt.plot(epochs, model_data[history_key], linestyle=linestyle, color=colors[model_name], 
                         #label=f"{model_name} - {label_name}" if i == 0 else None)  # Show legend only in first plot
                         label=f"{model_name} - {label_name}")  # Show legend only in first plot

    plt.xlabel("Epochs")
    plt.ylabel(metric_label)
    plt.title(f"Comparison of {metric_label} for GraphRec and MLP-Based Models")

    #if i == 0:  # Only include legend in the first plot
    plt.legend()

    plt.grid(True)
    plt.savefig(plot_filenames[metric_key])  # Save figure
    plt.close()
