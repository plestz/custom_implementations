import torch
import torch.nn as nn

import numpy as np

from src.encoder import Encoder
from src.decoder import Decoder
from src.embedding import Embedding, CustomEmbedding
from src.utils import pad_batch_to_longest_seq_len

from abc import ABC, abstractmethod

class Transformer(ABC, nn.Module):
    """
    An abstract class of a Transformer. 

    Subclasses are EncoderDecoder, Encoder-only, and Decoder-only varieties.

    Classes are loosely based off of PyTorch's argument structure.
    """
    def __init__(self,
                 embeddings: CustomEmbedding,
                 vocab_size: int,
                 d_model: int = 512,
                 num_attention_heads: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: nn.Module = nn.ReLU(),
                 layer_norm_epsilon: float = 1e-5,
                 max_context_window: int = 1024,
                 use_pre_lnorm: bool = False
        ):
        """
        Transformer intitializer.

        Args:
            embeddings - A container for all of the word embeddings in the vocabulary (including PAD, SOS, EOS)
            vocab_size - The size of the vocabulary that this transformer will process
            d_model - The embedding & hidden dimension of the transformer
            num_attention_heads - The number of attention heads
            dim_feedforward - The feed-forward layer's hidden dimension 
            dropout - The dropout percentage for the network
            activation - The activation function to use in the hidden linear layer at the end of each encoder/decoder block
            layer_norm_epsilon - The epsilon (numerical stability) to use for each LayerNorm layer
            max_context_window - The maximum context window that this transformer will be used to process
            use_pre_lnorm - A flag to determine whether to use pre-norm (if set to true) or post-norm (otherwise)
        """
        super().__init__()

        if d_model % num_attention_heads != 0:
            raise ValueError(f'd_model must be evenly divisible by the number of attention heads, but {d_model} % {num_attention_heads} = {d_model % num_attention_heads}.')

        self.embeddings = embeddings
        self.vocab_size: int = vocab_size
        self.d_model: int = d_model
        self.num_attention_heads: int = num_attention_heads
        self.dim_feedforward: int = dim_feedforward
        self.dropout: float = dropout
        self.activation: nn.Module = activation
        self.layer_norm_epsilon: float = layer_norm_epsilon
        self.max_context_window = max_context_window
        self.use_pre_lnorm = use_pre_lnorm

        self.PAD_TOKEN_IDX = self.vocab_size - 3

        self.positional_encodings = self.get_all_positional_encodings().float()

    @abstractmethod
    def forward(self, x):
        """
        Abstract requirement to implement a method that performs full forward 
        pass through the model in subclasses.
        """
        pass

    def get_all_positional_encodings(self) -> torch.Tensor:
        """
        Produces all possible positional encodings according to the maximum
        context window and d_model, to be stored once at transformer creation.

        Returns:
            positional_encodings - All possible positional encodings in the shape
            of (seq, d_model)
        """
        positional_encodings = np.empty((self.max_context_window, self.d_model))

        positions = np.arange(self.max_context_window).reshape(-1, 1)

        dimensions = np.arange(self.d_model // 2)
        scaling_factor = np.power(10000, 2 * dimensions / self.d_model).reshape(1, -1)

        # Note: np.divide broadcasts two arrays of type [N, 1], [1, M] to [N, M]
        raw_encodings = np.divide(positions, scaling_factor)
        sin_encodings = np.sin(raw_encodings)
        cos_encodings = np.cos(raw_encodings)

        positional_encodings[np.ix_(positions.flatten(), dimensions * 2)] = sin_encodings
        positional_encodings[np.ix_(positions.flatten(), dimensions * 2 + 1)] = cos_encodings

        return torch.from_numpy(positional_encodings)

class EncoderDecoderTransformer(Transformer):
    """
    My custom implementation of an EncoderDecoder Transformer,
    based off of the 2017 research paper 'Attention Is All You Need'.
    """
    def __init__(self, 
                 embeddings: CustomEmbedding,
                 vocab_size: int,
                 d_model = 512, 
                 num_attention_heads = 8, 
                 num_encoder_layers = 6,
                 num_decoder_layers = 6,
                 dim_feedforward = 2048,
                 dropout = 0.1,
                 activation = nn.ReLU(),
                 layer_norm_epsilon = 1e-5,
                 max_context_window = 1024,
                 use_pre_lnorm: bool = False,
        ):
        """
        Encoder-Decoder Transformer intitializer.

        Args:
            embeddings - A container for all of the word embeddings in the vocabulary (including PAD, SOS, EOS)
            vocab_size - The size of the vocabulary that this transformer will process
            d_model - The embedding & hidden dimension of the transformer
            num_attention_heads - The number of attention heads
            num_encoder_layers - The number of sequential encoder blocks in the network
            num_decoder_layers - The number of sequential decoder blocks in the network
            dim_feedforward - The feed-forward layer's hidden dimension 
            dropout - The dropout percentage for the network
            activation - The activation function to use in the hidden linear layer at the end of each encoder/decoder block
            layer_norm_epsilon - The epsilon (numerical stability) to use for each LayerNorm layer
            max_context_window - The maximum context window that this transformer will be used to process
            use_pre_lnorm - A flag to determine whether to use pre-norm (if set to true) or post-norm (otherwise)
        """
        super().__init__(
            embeddings,
            vocab_size,
            d_model,
            num_attention_heads,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_epsilon,
            max_context_window,
            use_pre_lnorm
        )

        self.num_encoder_layers: int = num_encoder_layers
        self.num_decoder_layers: int = num_decoder_layers

        self.dropout_encoder_embedding = nn.Dropout(self.dropout)
        self.dropout_decoder_embedding = nn.Dropout(self.dropout)

        self.encoders = nn.ModuleList([Encoder(self.d_model, self.num_attention_heads, self.dim_feedforward, self.dropout, self.activation, self.layer_norm_epsilon, self.use_pre_lnorm) for _ in range(self.num_encoder_layers)])
        self.decoders = nn.ModuleList([Decoder(self.d_model, self.num_attention_heads, self.dim_feedforward, self.dropout, self.activation, self.layer_norm_epsilon, self.use_pre_lnorm) for _ in range(self.num_decoder_layers)])
        
        self.final_layer_norm = nn.LayerNorm(self.d_model, eps = self.layer_norm_epsilon)
        self.vocab_linear = nn.Linear(self.d_model, self.vocab_size)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Performs one full forward pass through the transformer. Takes
        the padded batch of input sequences and outputs the probability distribution
        vector for the next-most likely word to appear after the target(s).
        
        Note that this is the classical Transformer architecture from the 2017 research 
        paper 'Attention Is All You Need', so there is an encoder and decoder.

        Args:
            source - Encoder input batch. A 2D tensor of shape (batch size = num_sequences, sequence length (padded to max)).
            target - Decoder input batch. A 2D tensor of shape (batch size = num_sequences, sequence length (padded to max)).

        Returns:
            next_word_probs - For each seq_i in the layer-normalized decoder_output, 
            produces the next_word_probabilities corresponding to the next most likely word.
        """
        # Encoder Block
        encoder_output, source_pad_mask = self.encode(source)

        # Decoder Block
        decoder_output, _ = self.decode(target, encoder_output, source_pad_mask)

        # Project into Vocabulary (+ Final LayerNorm)
        logits = self.project_into_vocab(decoder_output)

        return logits
    
    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one full forward pass through the *encoder* block of the Transformer.

        Takes the padded batch of input sequences and outputs a [B,S,D] encoded
        representation Tensor of the source context (bi-directional).

        Args:
            source - Encoder input batch. A 2D tensor of shape (batch size = num_sequences, sequence length (padded to max)).

        Returns:
            encoder_output - The output of the entire encoder block after bi-directional embedding contextualization
            source_pad_mask - The padding mask pertaining to the entire source batch
        """
        # At this point, source = (# sequences, unaltered sequence padded up to max_seq_len)
        source_pad_mask = (source != self.PAD_TOKEN_IDX).bool()

        # The goal is turn the "batch" of (1, sequence)'s into the corresponding (3D) batch of (seq_i_len, d_model)
        source_embedding: torch.Tensor = self.embeddings(source)

        source_batch_size, source_max_sequence_len = source.size()

        assert source_max_sequence_len <= self.max_context_window
        assert source_embedding.size() == (source_batch_size, source_max_sequence_len, self.d_model)

        # Encoders (Sequential Processing)
        encoder_input = self.dropout_encoder_embedding(source_embedding + self.positional_encodings[:source_max_sequence_len].unsqueeze(0))

        encoder_output = encoder_input
        for i in range(self.num_encoder_layers):
            encoder_output = self.encoders[i](encoder_output, source_pad_mask)

        return encoder_output, source_pad_mask
    
    def decode(self, target: torch.Tensor, encoder_output: torch.Tensor, source_pad_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs one full forward pass through the *decoder* block of the Transformer.

        Takes the encoder input and outputs a [B,S,D] Tensor that encapsulates, for every
        target token embedding, (1) the entire source context, and (2) its left-context
        within the batch.

        Args:
            target - Decoder input batch. A 2D tensor of shape (batch size = num_sequences, sequence length (padded to max)).
            encoder_output - The output of the entire encoder block after bi-directional embedding contextualization
            source_pad_mask - The padding mask pertaining to the entire source batch

        Returns:
            decoder_output - The output of the entire decoder block after contextualization (see function description).
            target_pad_mask - The padding mask pertaining to the entire target batch
        """
        # At this point, target = (# sequences, [SOS] + desired sequence padding up to max_seq_len + 1)
        target_pad_mask = (target != self.PAD_TOKEN_IDX).bool()

        # The goal is turn the "batch" of (1, sequence)'s into the corresponding (3D) batch of (seq_i_len, d_model)
        target_embedding: torch.Tensor = self.embeddings(target)

        target_batch_size, target_max_sequence_len = target.size()

        assert target_max_sequence_len <= self.max_context_window
        assert target_embedding.size() == (target_batch_size, target_max_sequence_len, self.d_model)

        encoder_K, encoder_V = encoder_output, encoder_output

        # Decoders (Sequential Processing)
        decoder_input = self.dropout_decoder_embedding(target_embedding + self.positional_encodings[:target_max_sequence_len].unsqueeze(0))

        decoder_output = decoder_input
        for i in range(self.num_decoder_layers):
            decoder_output = self.decoders[i](decoder_output, encoder_K, encoder_V, target_pad_mask, source_pad_mask)

        return decoder_output, target_pad_mask
    
    def project_into_vocab(self, decoder_output: torch.Tensor) -> torch.Tensor:
        """
        Projects the output of the decoder into the vocabulary space with
        corresponding non-negative logits for every token.

        Args:
            decoder_output - The output of the entire decoder block after contextualization.

        Returns:
            logits - The logits corresponding to the raw, unnormalized probabilities of each token in the vocabulary
        """
        # Final layer norm before projection for stabilization
        norm_output = self.final_layer_norm(decoder_output)

        # Project to Vocabulary Space
        logits = self.vocab_linear(norm_output)

        return logits