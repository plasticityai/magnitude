
from __future__ import absolute_import
#typing
import torch

from allennlp.common import Params
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.modules.elmo import Elmo
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.data import Vocabulary


class ElmoTokenEmbedder(TokenEmbedder):
    u"""
    Compute a single layer of ELMo representations.

    This class serves as a convenience when you only want to use one layer of
    ELMo representations at the input of your network.  It's essentially a wrapper
    around Elmo(num_output_representations=1, ...)

    Parameters
    ----------
    options_file : ``str``, required.
        An ELMo JSON options file.
    weight_file : ``str``, required.
        An ELMo hdf5 weight file.
    do_layer_norm : ``bool``, optional.
        Should we apply layer normalization (passed to ``ScalarMix``)?
    dropout : ``float``, optional.
        The dropout value to be applied to the ELMo representations.
    requires_grad : ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    projection_dim : ``int``, optional
        If given, we will project the ELMo embedding down to this dimension.  We recommend that you
        try using ELMo with a lot of dropout and no projection first, but we have found a few cases
        where projection helps (particulary where there is very limited training data).
    vocab_to_cache : ``List[str]``, optional, (default = 0.5).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, the ElmoTokenEmbedder expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    """
    def __init__(self,
                 options_file     ,
                 weight_file     ,
                 do_layer_norm       = False,
                 dropout        = 0.5,
                 requires_grad       = False,
                 projection_dim      = None,
                 vocab_to_cache            = None)        :
        super(ElmoTokenEmbedder, self).__init__()

        self._elmo = Elmo(options_file,
                          weight_file,
                          1,
                          do_layer_norm=do_layer_norm,
                          dropout=dropout,
                          requires_grad=requires_grad,
                          vocab_to_cache=vocab_to_cache)
        if projection_dim:
            self._projection = torch.nn.Linear(self._elmo.get_output_dim(), projection_dim)
        else:
            self._projection = None

    def get_output_dim(self):
        return self._elmo.get_output_dim()

    def forward(self, # pylint: disable=arguments-differ
                inputs              ,
                word_inputs               = None)                :
        u"""
        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, optional.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.

        Returns
        -------
        The ELMo representations for the input sequence, shape
        ``(batch_size, timesteps, embedding_dim)``
        """
        elmo_output = self._elmo(inputs, word_inputs)
        elmo_representations = elmo_output[u'elmo_representations'][0]
        if self._projection:
            projection = self._projection
            for _ in range(elmo_representations.dim() - 2):
                projection = TimeDistributed(projection)
            elmo_representations = projection(elmo_representations)
        return elmo_representations

    # Custom vocab_to_cache logic requires a from_params implementation.
    @classmethod
    def from_params(cls, vocab            , params        )                       :  # type: ignore
        # pylint: disable=arguments-differ
        params.add_file_to_archive(u'options_file')
        params.add_file_to_archive(u'weight_file')
        options_file = params.pop(u'options_file')
        weight_file = params.pop(u'weight_file')
        requires_grad = params.pop(u'requires_grad', False)
        do_layer_norm = params.pop_bool(u'do_layer_norm', False)
        dropout = params.pop_float(u"dropout", 0.5)
        namespace_to_cache = params.pop(u"namespace_to_cache", None)
        if namespace_to_cache is not None:
            vocab_to_cache = list(vocab.get_token_to_index_vocabulary(namespace_to_cache).keys())
        else:
            vocab_to_cache = None
        projection_dim = params.pop_int(u"projection_dim", None)
        params.assert_empty(cls.__name__)
        return cls(options_file=options_file,
                   weight_file=weight_file,
                   do_layer_norm=do_layer_norm,
                   dropout=dropout,
                   requires_grad=requires_grad,
                   projection_dim=projection_dim,
                   vocab_to_cache=vocab_to_cache)

ElmoTokenEmbedder = TokenEmbedder.register(u"elmo_token_embedder")(ElmoTokenEmbedder)
