import torch
import torch.nn.functional as F
import torch.nn as nn


def hf_loss_fn(batch, outputs):
    loss = outputs.loss
    return loss


def apply_ce(logits, labels):
    """Apply cross-entropy loss with tensors reshaped"""
    # Reshape logits and labels
    batch_size, seq_len, num_classes = logits.shape
    # Shape [batch_size * sequence_length, num_classes]
    logits = logits.reshape(-1, num_classes)
    # Shape [batch_size * sequence_length]
    labels = labels.reshape(-1)
    return F.cross_entropy(logits, labels, ignore_index=-100)


def twotasks_ce_loss_fn(batch, outputs):
    half_batch_len = len(batch["labels"]) // 2
    title_logits = outputs["logits"][0:half_batch_len]
    keywords_logits = outputs["logits"][half_batch_len:]
    title_labels = batch["labels"][0:half_batch_len]
    keywords_labels = batch["labels"][half_batch_len:]

    title_loss = apply_ce(title_logits, title_labels)
    keywords_loss = apply_ce(keywords_logits, keywords_labels)

    combined_loss = title_loss + keywords_loss
    return combined_loss


def twotasks_ce_eisl_loss_fn(batch, outputs):
    def config_ngram_list(ngram, output_length):
        ngram_list = set()
        for n in ngram:
            if n > 0:
                if n <= output_length:
                    ngram_list.add(n)
            else:
                real_n = output_length + n
                if 0 < real_n:
                    ngram_list.add(real_n)
        if ngram_list:
            ngram_list = list(ngram_list)
        else:
            ngram_list = [output_length]
        return ngram_list

    def batch_log_EISL_cnn(
        decoder_outputs, target_idx, ngram_list, pad=1, weight_list=None
    ):
        """
        decoder_outputs: [batch_size, output_len, vocab_size]
            - matrix with probabilityes  -- log probs
        target_variable: [batch_size, target_len]
            - reference batch
        ngram_list: int or List[int]
            - n-gram to consider
        pad: int
            the idx of "pad" token
        weight_list : List
            corresponding weight of ngram

        NOTE: output_len == target_len
        """
        batch_size, output_len, vocab_size = decoder_outputs.size()
        _, tgt_len = target_idx.size()

        if type(ngram_list) == int:
            ngram_list = [ngram_list]
        if ngram_list[0] <= 0:
            ngram_list[0] = output_len
        if weight_list is None:
            weight_list = [1.0 / len(ngram_list)] * len(ngram_list)

        decoder_outputs = torch.relu(decoder_outputs + 20) - 20  # Filter out the
        # [batch_size, output_len, target_len]
        index = target_idx.unsqueeze(1).expand(-1, output_len, tgt_len)
        # [batch, output_len, target_len]
        cost_nll = decoder_outputs.gather(dim=2, index=index)
        # [batch, 1, output_len, target_len]
        cost_nll = cost_nll.unsqueeze(1)
        sum_gram = torch.tensor([0.0], dtype=cost_nll.dtype, device=cost_nll.device)
        for cnt, ngram in enumerate(ngram_list):
            # out: [batch, 1, output_len, target_len]
            # eye_filter: [1, 1, ngram, ngram]
            eye_filter = torch.eye(ngram).view([1, 1, ngram, ngram]).cuda()
            assert ngram <= decoder_outputs.size()[1]
            # term: [batch, 1, output_len - ngram + 1, target_len - ngram + 1]
            term = nn.functional.conv2d(cost_nll, eye_filter) / ngram
            # maybe dim should be 2, but sometime 1 is better
            gum_tmp = F.gumbel_softmax(term.squeeze_(1), tau=1, dim=1)
            term = term.mul(gum_tmp).sum(1).mean(1)
            sum_gram += weight_list[cnt] * term.sum()
        loss = -sum_gram / batch_size
        return loss

    half_batch_len = len(batch["labels"]) // 2
    title_logits = outputs["logits"][0:half_batch_len]
    keywords_logits = outputs["logits"][half_batch_len:]
    title_labels = batch["labels"][0:half_batch_len]
    keywords_labels = batch["labels"][half_batch_len:]

    title_loss = apply_ce(title_logits, title_labels)

    log_probs = F.log_softmax(keywords_logits, dim=-1)
    ngram_list = config_ngram_list(None, output_length=outputs.size(1))
    keywords_loss = batch_log_EISL_cnn(
        log_probs, keywords_labels, ngram_list=ngram_list
    )

    combined_loss = title_loss + keywords_loss
    return combined_loss
