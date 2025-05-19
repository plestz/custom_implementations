import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

def run_train_epoch(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, calculate_sequence_accuracy: bool = False, calculate_token_accuracy: bool = False):
    """
    Runs one training epoch (processing the entire training dataset once).
    
    Uses Teacher Forcing to train token-to-token mapping quality without cascading errors and for parallelization.

    Args:
        dataloader - The dataloader to process the dataset in BATCH_SIZE batches
        model - The Encoder-Decoder that is being trained
        loss_fn - The loss function to calculate the model's correctness
        optimizer - The optimizer to improve the model's weights
        calculate_sequence_accuracy - A flag to mark whether sequence-level correctness should be tracked
        calculate_token_accuracy - A flag to mark whether token-level correctness should be tracked
    """
    model.train()

    num_sequences = len(dataloader.dataset)
    num_tokens = 0

    epoch_loss = 0.0
    total_correct_sequences = 0
    total_correct_tokens = 0

    for (source, target), label in tqdm(dataloader):

        # FORWARD
        pred_logits = model(source, target)

        # pred_logits.shape: [batch_size, seq_len, vocab_size]
        # label.shape: [batch_size, seq_len]

        # CrossEntropyLoss (loss_fn) only takes 2D predictions (n_batch * seq_len, vocab_size) and 1D labels (n_batch * seq_len)
        batch_loss = loss_fn(pred_logits.view(-1, pred_logits.size(-1)), label.view(-1))

        # LOG
        with torch.no_grad():
            epoch_loss += batch_loss.item()

            predictions = torch.argmax(pred_logits, dim = -1) # predictions.shape: [batch_size, seq_len]
            match_matrix = torch.eq(predictions, label)

            if calculate_sequence_accuracy:
                num_correct_sequences = torch.all(match_matrix, dim = 1).sum()
                total_correct_sequences += num_correct_sequences.item()

            if calculate_token_accuracy:
                num_correct_tokens = match_matrix.sum()      
                total_correct_tokens += num_correct_tokens.item()

                num_tokens += torch.numel(label)

        # BACKWARD
        batch_loss.backward()

        # OPTIMIZE
        optimizer.step()
        optimizer.zero_grad()

    average_epoch_loss = epoch_loss / num_sequences
    average_epoch_sequence_accuracy = total_correct_sequences / num_sequences if calculate_sequence_accuracy else None
    average_epoch_token_accuracy = total_correct_tokens / num_tokens if calculate_token_accuracy else None

    return average_epoch_loss, average_epoch_sequence_accuracy, average_epoch_token_accuracy