import math
from typing import Tuple, List
import numpy as np

def compute_bits_per_byte(loss_nats: float, num_tokens: int, num_bytes: int) -> float:
    """
    Computes bits-per-byte (BPB) for tokenizer-independent language modeling evaluation:
    BPB = (loss_nats / ln(2)) * (num_tokens / num_bytes)
    """
    if num_bytes <= 0 or num_tokens <= 0:
        return 0.0
    bits_per_token = loss_nats / math.log(2.0)
    return round(bits_per_token * (num_tokens / num_bytes), 4)

def compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Computes Matthews Correlation Coefficient (MCC) for binary classification (e.g. CoLA).
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    """
    numerator = float(tp * tn - fp * fn)
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if denominator == 0.0:
        return 0.0
    return round(numerator / denominator, 4)

def compute_confidence_interval_95(accuracy: float, sample_size: int) -> Tuple[float, float]:
    """
    Computes standard 95% Wald confidence interval for classification accuracy.
    Margin = 1.96 * sqrt(p * (1 - p) / N)
    """
    if sample_size <= 0:
        return (0.0, 0.0)
    p = accuracy / 100.0 if accuracy > 1.0 else accuracy
    margin = 1.96 * math.sqrt(max(0.0, p * (1.0 - p)) / sample_size)
    lower = max(0.0, (p - margin) * 100.0)
    upper = min(100.0, (p + margin) * 100.0)
    return (round(lower, 2), round(upper, 2))

def compute_lambada_last_word_match(prediction: str, ground_truth_last_word: str) -> bool:
    """
    Evaluates LAMBADA last-word prediction accuracy.
    """
    pred_words = prediction.strip().split()
    if not pred_words:
        return False
    pred_last = pred_words[-1].strip(".,!?:;\"'()[]{}").lower()
    target_last = ground_truth_last_word.strip(".,!?:;\"'()[]{}").lower()
    return pred_last == target_last

