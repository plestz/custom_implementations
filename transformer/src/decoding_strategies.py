import torch
import torch.nn as nn

def greedy_decode(source: torch.Tensor, model: nn.Module, vocab_size: int, special_token_idxs: dict, max_context_window: int) -> torch.Tensor:
    """
    Designed to do autoregressive inference in an Encoder-Decoder transformer.

    This greedy decoder always predicts the vocabulary token corresponding to the highest logit.

    Takes the source sequence and the Encoder-Decoder to produce a predicted
    sequence starting from <SOS>.

    Note: This function *can* handle batches of sequences.

    Args:
        source - The source sequence to be passed to the Transformer's encoder block.
        model - The Encoder-Decoder transformer over which to greedy decode
        vocab_size - The size of the embedding vocabulary
        special_token_idxs - The indices corresponding to the SOS, EOS, and PAD tokens
        max_context_window - The maximum context window that the Transformer can process

    Returns:
        target - The batch of predicted sequences corresponding to the input sources.
        target_logits - (# batch elements =) batch_size (# rows =) seq_len (# cols =) vocab-dimensional vectors, each of which
        corresponds to the set of logits on a particular inference step within a given sequence.
    """
    batch_size = source.size(dim = 0)

    VOCAB_SIZE = vocab_size
    MAX_CONTEXT_WINDOW = max_context_window
    SOS_TOKEN_IDX = special_token_idxs['SOS_TOKEN_IDX']
    EOS_TOKEN_IDX = special_token_idxs['EOS_TOKEN_IDX']
    PAD_TOKEN_IDX = special_token_idxs['PAD_TOKEN_IDX']

    encoder_output, source_pad_mask = model.encode(source)

    # target will contain num_batch sequences of indices that are the predicted next-words for each batch element
    target = torch.full((batch_size, 1), SOS_TOKEN_IDX) # target.shape: [batch_size, num_loops_complete - 1]
    target_logits = torch.zeros((batch_size, 1, VOCAB_SIZE))

    finished = torch.full((batch_size, ), False)

    while not finished.all() and target.size(dim = 1) <= MAX_CONTEXT_WINDOW:

        decoder_output, _ = model.decode(target, encoder_output, source_pad_mask)
        pred_logits = model.project_into_vocab(decoder_output) # pred_logits.shape: [batch_size, seq_len, vocab_size]

        last_row_pred_logits = pred_logits[:, -1, :] # last_row_pred_logits.shape == [batch_size, vocab_size]

        # Track next-word logits for loss_fn later.
        target_logits = torch.concat((target_logits, last_row_pred_logits.unsqueeze(1)), dim = 1)

        predictions = torch.argmax(last_row_pred_logits, dim = -1) # predictions.shape: [batch_size]

        # For any finished sequences (i.e. previous EOS-producers), force their prediction from this round to be a pad.
        predictions[finished] = PAD_TOKEN_IDX

        # Mark any additional sequences that just produced an EOS as finished.
        finished |= predictions == EOS_TOKEN_IDX

        target = torch.concat((target, predictions.reshape(-1, 1)), dim = 1) # target.shape: [batch_size, num_loops_complete]

    return target, target_logits[:, 1:, :]