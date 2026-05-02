import math
import torch


def perplexity(loss: float) -> float:
    return math.exp(loss)


def token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item()
