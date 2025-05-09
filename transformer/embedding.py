import torch
import torch.nn as nn
# import torchtext
# from torchtext.vocab import GloVe, FastText, Vectors
# from torchtext.data.utils import get_tokenizer

class Embedding:
    """
    An embedding class for any indexable-by-word embedding list.
    """
    def __init__(self, embeddings):
        """
        Initializes Embedding, compatible with any indexable-by-word embedding list.

        Args:
            embeddings - Word embeddings, indexable by word.
        """
        self.embeddings = embeddings

    def __getitem__(self, key):
        """
        Provides specified item(s) from word embeddings.

        Args:
            key -- Either a single word (string) or list of words (list of strings)
            whose n_embedding(s) will be provided. Returned shape is (n_embedding) if
            key is a single word, and (n_words, n_embedding) if key is a list.
        """
        if isinstance(key, str):
            return self.embeddings[key]
        elif isinstance(key, list):
            return torch.cat([self.embeddings[word].unsqueeze(0) for word in key], dim = 0)
        else:
            raise ValueError(f'Unsupported index key type: {key}.')
        
    def __call__(self, indices: torch.Tensor):
        return self.embeddings(indices)
        


# class PretrainedEmbedding(Embedding):
#     """
#     An embedding class for pre-trained torchtext embeddings.

#     Subclasses Embedding for instance variable and embedding access utilities.
#     """

#     def __init__(self, embeddings: Vectors):
#         """
#         Initializes PretrainedEmbedding with torchtext.vocab pretrained word embeddings.

#         Args:
#             embeddings: torchtext.vocab pretrained word embeddings.
#         """
#         super().__init__(embeddings)

class CustomEmbedding(Embedding):
    """
    An embedding class for new, custom embeddings that will be *trained*.
    
    Subclasses Embedding for instance variable and embedding access utilities.
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initializes new embeddings according to vocab_size, d_model dimensions.

        Args:
            vocab_size: The total size of the vocabulary of the model
            d_model: The embedding dimension of the model
        """
        super().__init__(nn.Embedding(vocab_size, d_model))

# if __name__ == '__main__':
    
#     glove = GloVe(name = '6B', dim = 100, cache = 'embeddings/.vector_cache')

#     tokenizer = get_tokenizer(tokenizer = 'basic_english')
#     tokens = tokenizer('I love transformers')

#     assert tokens == ['i', 'love', 'transformers']

#     input_embeddings = PretrainedEmbedding(glove)

#     assert torch.all(input_embeddings[tokens][0] == input_embeddings['i'])