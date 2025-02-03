import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def get_link_prediction_metrics(predicts, labels):
    """
    Computes Accuracy@1 and Pairwise Ranking Accuracy for link prediction.

    :param predicts: Tensor of shape (batch_size, 5), raw scores for [positive, 4 negatives].
    :param labels: Tensor of shape (batch_size, 5), where 1 represents the positive sample, 0 for negatives.
    :return: Dictionary containing Accuracy@1 and Pairwise Ranking Accuracy.
    """
    # Convert to CPU for evaluation
    predicts, labels = predicts.detach().cpu(), labels.detach().cpu()

    # Compute Accuracy @1 (How often the positive is ranked highest)
    accuracy = (predicts.argmax(dim=-1) == labels.argmax(dim=-1)).float().mean().item()

    # **Pairwise Ranking Accuracy Calculation**
    # Compare the positive score with all four negatives for each row
    batch_size = predicts.shape[0]

    # Get scores of the positive item (batch_size, 1)
    positive_scores = predicts.gather(1, labels.argmax(dim=-1, keepdim=True))

    # Get scores of negative items (batch_size, 4)
    negative_scores = torch.where(labels == 0, predicts, torch.tensor(float('-inf')).to(predicts.device))

    # Compute how often positive scores are greater than negatives
    pairwise_correct = (positive_scores > negative_scores).float()

    # Average over all pairwise comparisons
    pairwise_acc = pairwise_correct.mean().item()

    return {
        "accuracy": accuracy,
        "pairwise_acc": pairwise_acc
    }