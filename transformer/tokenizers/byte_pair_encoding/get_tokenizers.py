from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

def train_and_save_tokenizer_for(file_path: str):

    tokenizer = ByteLevelBPETokenizer()

    tokenizer.train(
        files = [file_path],
        vocab_size = 10_000,
        min_frequency = 2,
        special_tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>', '<MASK>'],
    )

    tokenizer.save("../trained_tokenizers/SAMSum_BPE/tokenizer.json")
    tokenizer.save_model("../trained_tokenizers/SAMSum_BPE")

def load_tokenizer_from(file_path: str):

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file = '../trained_tokenizers/SAMSum_BPE/tokenizer.json',
        vocab_file = '../trained_tokenizers/SAMSum_BPE/vocab.json',
        merges_file = '../trained_tokenizers/SAMSum_BPE/merges.txt',
        bos_token = '<SOS>',
        eos_token = '<EOS>',
        pad_token = '<PAD>',
        unk_token = '<UNK>',
        mask_token = '<MASK>'
    )

    return tokenizer

if __name__ == '__main__':

    train_and_save_tokenizer_for('../../data/SAMSum/train_summary_and_dialogue.txt')