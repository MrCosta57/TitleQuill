import torch.nn.functional as F


def hf_loss_fn(batch, outputs):
    loss = outputs.loss
    return loss


def twotasks_ce_loss_fn(batch, outputs):

    def apply_ce(logits, labels):
        """Apply cross-entropy loss with tensors reshaped"""
        # Reshape logits and labels
        batch_size, seq_len, num_classes = logits.shape
        # Shape [batch_size * sequence_length, num_classes]
        logits = logits.reshape(-1, num_classes)
        # Shape [batch_size * sequence_length]
        labels = labels.reshape(-1)
        return F.cross_entropy(logits, labels)

    print("INSIDE LOSS FUNCTION")
    print("Batch")
    print({k: v.shape for k, v in batch.items()})

    half_batch_len = len(batch["labels"]) // 2

    title_logits = outputs["logits"][0:half_batch_len]
    keywords_logits = outputs["logits"][half_batch_len:]
    title_labels = batch["labels"][0:half_batch_len]
    keywords_labels = batch["labels"][half_batch_len:]

    title_loss = apply_ce(title_logits, title_labels)
    keywords_loss = apply_ce(keywords_logits, keywords_labels)

    combined_loss = title_loss + keywords_loss
    return combined_loss
