# Drop-in patch for anomalyDetector.py
# Replace your existing `get_precision_recall` with this version.

import torch

def get_precision_recall(args, score, num_samples=1000, beta=1.0, label=None, predicted_score=None):
    """
    Computes precision, recall, and F-beta over a sweep of thresholds.

    Args:
        args: argparse.Namespace with `.device`
        score (Tensor): anomaly scores for the test sequence, shape [T] or [T, 1]
        num_samples (int): number of thresholds to sweep
        beta (float): beta parameter for F-beta
        label (Tensor): binary ground-truth labels, shape [T] or [T, 1]
        predicted_score (ndarray or Tensor, optional): compensation scores (same length as score)

    Returns:
        precision (Tensor), recall (Tensor), f_beta (Tensor) each of shape [num_samples]
    """
    device = getattr(args, 'device', 'cpu')

    # Ensure 1-D tensors on the right device
    score = torch.as_tensor(score, dtype=torch.float32, device=device).view(-1)
    if label is not None:
        label = torch.as_tensor(label, dtype=torch.float32, device=device).view(-1)
        # Binzarize labels in case they are not {0,1}
        label = (label > 0.5).float()

    if predicted_score is not None:
        predicted_score = torch.as_tensor(predicted_score, dtype=torch.float32, device=device).view(-1)
        # Simple compensation (optional): divide raw score by (predicted + eps)
        eps = 1e-8
        score = score / (predicted_score + eps)

    # Build a robust threshold vector in log-space without copy-construct warnings
    maximum = torch.max(score).detach()
    max_val = torch.clamp(torch.as_tensor(maximum, dtype=torch.float32, device=device), min=1e-8)
    th = torch.logspace(start=0.0, end=torch.log10(max_val).item(), steps=num_samples, device=device)

    precisions = []
    recalls = []
    f_betas = []

    # Pre-compute positives
    if label is not None:
        TP_denom = torch.clamp(label.sum(), min=1.0)  # avoid divide-by-zero if no positives

    for t in th:
        pred = (score >= t).float()

        if label is None:
            # If no labels, we cannot compute PR; fill with NaNs
            precisions.append(torch.tensor(float('nan'), device=device))
            recalls.append(torch.tensor(float('nan'), device=device))
            f_betas.append(torch.tensor(float('nan'), device=device))
            continue

        TP = (pred * label).sum()
        FP = (pred * (1.0 - label)).sum()
        FN = ((1.0 - pred) * label).sum()

        precision = TP / torch.clamp(TP + FP, min=1.0)
        recall = TP / TP_denom

        b2 = beta * beta
        f_beta = (1.0 + b2) * (precision * recall) / torch.clamp(b2 * precision + recall, min=1e-8)

        precisions.append(precision)
        recalls.append(recall)
        f_betas.append(f_beta)

    precision = torch.stack(precisions)
    recall = torch.stack(recalls)
    f_beta = torch.stack(f_betas)

    return precision, recall, f_beta
