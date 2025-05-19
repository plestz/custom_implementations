import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizerFast

from src.decoding_strategies import greedy_decode

from tqdm import tqdm

from typing import Union, Callable

def run_gold_validation_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, calculate_sequence_accuracy: bool = False, calculate_token_accuracy: bool = False):
    """
    Runs one validation epoch (processing the entire validation dataset once). 

    Uses Teacher Forcing (i.e. "gold") to evaluate token-to-token mapping quality and for parallelization.

    Args:
        dataloader - The dataloader to process the dataset in BATCH_SIZE batches
        model - The Encoder-Decoder that is being trained
        loss_fn - The loss function to calculate the model's correctness
        calculate_sequence_accuracy - A flag to mark whether sequence-level correctness should be tracked
        calculate_token_accuracy - A flag to mark whether token-level correctness should be tracked
    """
    model.eval()

    num_sequences = len(dataloader.dataset)
    num_tokens = 0

    epoch_loss = 0.0
    total_correct_sequences = 0
    total_correct_tokens = 0

    with torch.no_grad():
        
        for (source, target), label in tqdm(dataloader):
            
            # FORWARD
            pred_logits = model(source, target)
            batch_loss = loss_fn(pred_logits.view(-1, pred_logits.size(-1)), label.view(-1))

            # LOG
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

    average_epoch_loss = epoch_loss / num_sequences
    average_epoch_sequence_accuracy = total_correct_sequences / num_sequences if calculate_sequence_accuracy else None
    average_epoch_token_accuracy = total_correct_tokens / num_tokens if calculate_token_accuracy else None

    return average_epoch_loss, average_epoch_sequence_accuracy, average_epoch_token_accuracy

def run_autoregressive_validation_loop(dataloader: DataLoader, model: nn.Module, vocab_map: Union[dict, Callable], special_token_idxs: dict, max_context_window: int):
    """
    Runs one autoregressive validation epoch (processing the entire validation dataset once). 

    Args:
        dataloader - The dataloader to process the dataset in BATCH_SIZE batches
        model - The Encoder-Decoder that is being trained
        vocab_map - Maps token vocabulary indices to the corresponding token value
        special_token_idxs - The indices corresponding to the SOS, EOS, and PAD tokens
        max_context_window - The maximum context window that the Transformer can process
    """
    model.eval()

    correct_sequences = 0
    incorrect_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        
        for (source, _), label in tqdm(dataloader):

            # FORWARD
            pred_indices, pred_logits = greedy_decode(source, model, len(vocab_map), special_token_idxs, max_context_window)

            np_source_indices = source.numpy().copy()
            np_pred_target_indices = pred_indices.numpy().copy()

            if isinstance(vocab_map, dict):
                token_values = np.array(list(vocab_map.values()))
                predicted_source_tokens = token_values[np_source_indices]
                predicted_target_tokens = token_values[np_pred_target_indices]
            elif isinstance(vocab_map, Callable):
                predicted_source_tokens = vocab_map(np_source_indices)
                predicted_target_tokens = vocab_map(np_pred_target_indices)
            else:
                raise TypeError('Vocab map not passed with valid type.')

            for s, t in zip(predicted_source_tokens, predicted_target_tokens):
                source_end_index = np.argmax(s == '<PAD>') if '<PAD>' in s else len(s)
                target_end_index = np.argmax(t == '<EOS>')
                if np.array_equal(np.sort(s[:source_end_index]), t[1:target_end_index]):
                    correct_sequences += 1
                else:
                    incorrect_sequences += 1
                    print(f'Incorrect Sequence {incorrect_sequences}:')
                    print(np.sort(s[:source_end_index]))
                    print(t[1:target_end_index])
                    print(f'{'Source:':<20} {s}\n{'Predicted Target:':<20} {t}', end = '\n\n')

            total_sequences += predicted_target_tokens.shape[0]

    return correct_sequences / total_sequences

def temp_run_autoregressive_validation_loop(tokenizer: PreTrainedTokenizerFast, dataloader: DataLoader, model: nn.Module, vocab_size: int, vocab_map: dict, special_token_idxs: dict, max_context_window: int):
    """
    Runs one autoregressive validation epoch (processing the entire validation dataset once). 

    Args:
        tokenizer - The pretrained tokenizer to use to decode greedy sequences
        dataloader - The dataloader to process the dataset in BATCH_SIZE batches
        model - The Encoder-Decoder that is being trained
        vocab_size - The size of the vocabulary
        vocab_map - Maps token vocabulary indices to the corresponding token value
        special_token_idxs - The indices corresponding to the SOS, EOS, and PAD tokens
        max_context_window - The maximum context window that the Transformer can process
    """
    model.eval()

    with torch.no_grad():
        
        for (source, _), label in tqdm(dataloader):

            # FORWARD
            pred_indices, pred_logits = greedy_decode(source, model, vocab_size, special_token_idxs, max_context_window)

            np_source_indices = source.numpy().copy()
            np_pred_target_indices = pred_indices.numpy().copy()

            # token_values = np.array(list(vocab_map.values()))
            # predicted_source_tokens = token_values[np_source_indices]
            # predicted_target_tokens = token_values[np_pred_target_indices]

            for s, t in zip(np_source_indices, np_pred_target_indices):
                decoded_source = tokenizer.decode(s)
                decoded_predicted_target = tokenizer.decode(t)
                print(f'{'Source:':<20} {decoded_source}\n{'Predicted Target:':<20} {decoded_predicted_target}', end = '\n\n')

            break