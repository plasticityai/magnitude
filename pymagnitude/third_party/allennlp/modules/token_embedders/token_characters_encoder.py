
from __future__ import absolute_import
import torch

from allennlp.common import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


class TokenCharactersEncoder(TokenEmbedder):
    u"""
    A ``TokenCharactersEncoder`` takes the output of a
    :class:`~allennlp.data.token_indexers.TokenCharactersIndexer`, which is a tensor of shape
    (batch_size, num_tokens, num_characters), embeds the characters, runs a token-level encoder, and
    returns the result, which is a tensor of shape (batch_size, num_tokens, encoding_dim).  We also
    optionally apply dropout after the token-level encoder.

    We take the embedding and encoding modules as input, so this class is itself quite simple.
    """
    def __init__(self, embedding           , encoder                , dropout        = 0.0)        :
        super(TokenCharactersEncoder, self).__init__()
        self._embedding = TimeDistributed(embedding)
        self._encoder = TimeDistributed(encoder)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x

    def get_output_dim(self)       :
        return self._encoder._module.get_output_dim()  # pylint: disable=protected-access

    def forward(self, token_characters              )                :  # pylint: disable=arguments-differ
        mask = (token_characters != 0).long()
        return self._dropout(self._encoder(self._embedding(token_characters), mask))

    # The setdefault requires a custom from_params
    @classmethod
    def from_params(cls, vocab            , params        )                            :  # type: ignore
        # pylint: disable=arguments-differ
        embedding_params         = params.pop(u"embedding")
        # Embedding.from_params() uses "tokens" as the default namespace, but we need to change
        # that to be "token_characters" by default.
        embedding_params.setdefault(u"vocab_namespace", u"token_characters")
        embedding = Embedding.from_params(vocab, embedding_params)
        encoder_params         = params.pop(u"encoder")
        encoder = Seq2VecEncoder.from_params(encoder_params)
        dropout = params.pop_float(u"dropout", 0.0)
        params.assert_empty(cls.__name__)
        return cls(embedding, encoder, dropout)

TokenCharactersEncoder = TokenEmbedder.register(u"character_encoding")(TokenCharactersEncoder)
