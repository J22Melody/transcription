from typing import List

import torch
import numpy as np
from sklearn.metrics import f1_score


def frame_accuracy(probs: torch.Tensor, gold: torch.Tensor) -> float:
    """
    probs: [sequence_length x number_of_classes(3)]
    gold: [sequence_length]
    """
    return float(torch.sum(gold == probs.argmax(dim=1)) / gold.shape[0])

def frame_f1(probs: torch.Tensor, gold: torch.Tensor) -> float:
    """
    probs: [sequence_length x number_of_classes(3)]
    gold: [sequence_length]
    """
    return f1_score(gold.numpy(), probs.argmax(dim=1).numpy(), average='macro')

def segment_percentage(segments: List[dict], segments_gold: List[dict]) -> float:
    """
    segments: [{'start': 1, 'end': 2}, ...]
    """
    return len(segments) / len(segments_gold) if len(segments_gold) > 0 else 0

def segment_IoU(segments: List[dict], segments_gold: List[dict], max_len=1000000) -> float:
    segments_v = np.zeros(max_len)
    for segment in segments:
        segments_v[segment['start']:segment['end']] = 1

    segments_gold_v = np.zeros(max_len)
    for segment in segments_gold:
        segments_gold_v[segment['start']:segment['end']] = 1

    intersection = np.logical_and(segments_v, segments_gold_v)
    union = np.logical_or(segments_v, segments_gold_v)
    return float(np.sum(intersection) / np.sum(union)) if np.sum(union) > 0 else 0