from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

def train_and_save_tokenizer_for(in_file_paths: list[str], 
                                 out_file_dir_path: str = None,
                                 include_vocab_and_merges: bool = False,
                                 vocab_size: int = 10_000, 
                                 min_frequency: int = 2, 
                                 special_tokens: list[str] = ['<SOS>', '<EOS>', '<PAD>', '<UNK>', '<MASK>']) -> ByteLevelBPETokenizer:
    """
    Creates, trains, and saves a low-level ByteLevelBPETokenizer's vocabulary given text files.

    Args:
        in_file_paths: The list of file paths to read vocabulary bytes from
        out_file_dir_path: The directory path to save the result files (if None, json merge results are not saved)
        include_vocab_and_merges: A flag for whether to include vocab.json and merges.txt in addition to tokenizer.json
        vocab_size: The size of the vocabulary to create (if possible)
        min_frequency: The minimum number of tokens of a certain merge required to add the joined version to the vocabulary
        special_tokens: The special tokens required for the tokenizer to train

    Returns:
        tokenizer: The trained ByteLevelBPE tokenizer
    """
    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files = in_file_paths,
        vocab_size = vocab_size,
        min_frequency = min_frequency,
        special_tokens = special_tokens
    )

    if out_file_dir_path and out_file_dir_path[-1] != '/':
        out_file_dir_path += '/'

    if out_file_dir_path:
        tokenizer.save(out_file_dir_path + 'tokenizer.json')
    
        if include_vocab_and_merges:
            tokenizer.save_model(out_file_dir_path)

    return tokenizer

def load_tokenizer_from(dir_path: str = None, model_max_length: int = 512) -> PreTrainedTokenizerFast:
    """
    Loads a low-level ByteLevelBPETokenizer from a directory,
    and wraps it in a PreTrainedTokenizerFast for high-level use.

    Args:
        dir_path: The location from which to read the tokenization guidelines
        model_max_length: The maximum token sequence length that the tokenizer can support

    Returns:
        pretrained_tokenizer: The PreTrainedTokenizerFast tokenizer object
    """
    
    if dir_path and dir_path[-1] != '/':
        dir_path += '/'

    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file = dir_path + 'tokenizer.json',
        model_max_length = model_max_length,
        add_prefix_space = True,
        bos_token = '<SOS>',
        eos_token = '<EOS>',
        pad_token = '<PAD>',
        unk_token = '<UNK>',
        mask_token = '<MASK>'
    )

    return pretrained_tokenizer