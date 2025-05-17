from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

def train_and_save_tokenizer_for(in_file_paths: list[str], 
                                 out_file_dir_path: str = None,
                                 include_vocab_and_merges: bool = False,
                                 vocab_size: int = 10_000, 
                                 min_frequency: int = 2, 
                                 special_tokens: list[str] = ['<SOS>', '<EOS>', '<PAD>', '<UNK>', '<MASK>']) -> ByteLevelBPETokenizer:
    """
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

def load_tokenizer_from(dir_path: str = None, tokenizer_obj: ByteLevelBPETokenizer = None, model_max_length: int = 512) -> PreTrainedTokenizerFast:
    """
    """
    if (dir_path and tokenizer_obj) or not (dir_path or tokenizer_obj):
        raise ValueError('Cannot provide both a directory path and a tokenizer object.')
    
    if dir_path and dir_path[-1] != '/':
        dir_path += '/'

    if dir_path:
        pretrained_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file = dir_path + 'tokenizer.json',
            model_max_length = model_max_length,
            bos_token = '<SOS>',
            eos_token = '<EOS>',
            pad_token = '<PAD>',
            unk_token = '<UNK>',
            mask_token = '<MASK>'
        )
    else:
        pretrained_tokenizer = PreTrainedTokenizerFast(
            tokenizer_obj = tokenizer_obj,
            model_max_length = model_max_length,
            add_prefix_space = True,
            bos_token = '<SOS>',
            eos_token = '<EOS>',
            pad_token = '<PAD>',
            unk_token = '<UNK>',
            mask_token = '<MASK>'
        )

    return pretrained_tokenizer