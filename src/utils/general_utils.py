import torch
import numpy as np
import random
import torch.nn.functional as F
import nltk


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def pad_tensor(tensor, pad_token_id):
    # Find the max length in the batch
    max_len = max(t.size(-1) for t in tensor)
    # Pad all tensors to the max length
    padded_tensors = [
        F.pad(t, (0, max_len - t.size(-1)), value=pad_token_id) for t in tensor
    ]
    return torch.stack(padded_tensors, dim=0)


def postprocess_validation_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # ROUGE expects a newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels
