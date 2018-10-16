
from __future__ import absolute_import
#typing
import warnings

import torch
#overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


class BasicTextFieldEmbedder(TextFieldEmbedder):
    u"""
    This is a ``TextFieldEmbedder`` that wraps a collection of :class:`TokenEmbedder` objects.  Each
    ``TokenEmbedder`` embeds or encodes the representation output from one
    :class:`~allennlp.data.TokenIndexer`.  As the data produced by a
    :class:`~allennlp.data.fields.TextField` is a dictionary mapping names to these
    representations, we take ``TokenEmbedders`` with corresponding names.  Each ``TokenEmbedders``
    embeds its input, and the result is concatenated in an arbitrary order.

    Parameters
    ----------

    token_embedders : ``Dict[str, TokenEmbedder]``, required.
        A dictionary mapping token embedder names to implementations.
        These names should match the corresponding indexer used to generate
        the tensor passed to the TokenEmbedder.
    embedder_to_indexer_map : ``Dict[str, List[str]]``, optional, (default = None)
        Optionally, you can provide a mapping between the names of the TokenEmbedders
        that you are using to embed your TextField and an ordered list of indexer names
        which are needed for running it. In most cases, your TokenEmbedder will only
        require a single tensor, because it is designed to run on the output of a
        single TokenIndexer. For example, the ELMo Token Embedder can be used in
        two modes, one of which requires both character ids and word ids for the
        same text. Note that the list of token indexer names is `ordered`, meaning
        that the tensors produced by the indexers will be passed to the embedders
        in the order you specify in this list.
    allow_unmatched_keys : ``bool``, optional (default = False)
        If True, then don't enforce the keys of the ``text_field_input`` to
        match those in ``token_embedders`` (useful if the mapping is specified
        via ``embedder_to_indexer_map``).
    """
    def __init__(self,
                 token_embedders                          ,
                 embedder_to_indexer_map                       = None,
                 allow_unmatched_keys       = False)        :
        super(BasicTextFieldEmbedder, self).__init__()
        self._token_embedders = token_embedders
        self._embedder_to_indexer_map = embedder_to_indexer_map
        for key, embedder in list(token_embedders.items()):
            name = u'token_embedder_%s' % key
            self.add_module(name, embedder)
        self._allow_unmatched_keys = allow_unmatched_keys

    #overrides
    def get_output_dim(self)       :
        output_dim = 0
        for embedder in list(self._token_embedders.values()):
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(self, text_field_input                         , num_wrapping_dims      = 0)                :
        if list(self._token_embedders.keys()) != list(text_field_input.keys()):
            if not self._allow_unmatched_keys:
                message = u"Mismatched token keys: %s and %s" % (unicode(list(self._token_embedders.keys())),
                                                                unicode(list(text_field_input.keys())))
                raise ConfigurationError(message)
        embedded_representations = []
        keys = sorted(self._token_embedders.keys())
        for key in keys:
            # If we pre-specified a mapping explictly, use that.
            if self._embedder_to_indexer_map is not None:
                tensors = [text_field_input[indexer_key] for
                           indexer_key in self._embedder_to_indexer_map[key]]
            else:
                # otherwise, we assume the mapping between indexers and embedders
                # is bijective and just use the key directly.
                tensors = [text_field_input[key]]
            # Note: need to use getattr here so that the pytorch voodoo
            # with submodules works with multiple GPUs.
            embedder = getattr(self, u'token_embedder_{}'.format(key))
            for _ in range(num_wrapping_dims):
                embedder = TimeDistributed(embedder)
            token_vectors = embedder(*tensors)
            embedded_representations.append(token_vectors)
        return torch.cat(embedded_representations, dim=-1)

    # This is some unusual logic, it needs a custom from_params.
    @classmethod
    def from_params(cls, vocab            , params        )                            :  # type: ignore
        # pylint: disable=arguments-differ,bad-super-call

        # The original `from_params` for this class was designed in a way that didn't agree
        # with the constructor. The constructor wants a 'token_embedders' parameter that is a
        # `Dict[str, TokenEmbedder]`, but the original `from_params` implementation expected those
        # key-value pairs to be top-level in the params object.
        #
        # This breaks our 'configuration wizard' and configuration checks. Hence, going forward,
        # the params need a 'token_embedders' key so that they line up with what the constructor wants.
        # For now, the old behavior is still supported, but produces a DeprecationWarning.

        embedder_to_indexer_map = params.pop(u"embedder_to_indexer_map", None)
        if embedder_to_indexer_map is not None:
            embedder_to_indexer_map = embedder_to_indexer_map.as_dict(quiet=True)
        allow_unmatched_keys = params.pop_bool(u"allow_unmatched_keys", False)

        token_embedder_params = params.pop(u'token_embedders', None)

        if token_embedder_params is not None:
            # New way: explicitly specified, so use it.
            token_embedders = dict((
                    name, TokenEmbedder.from_params(subparams, vocab=vocab))
                    for name, subparams in list(token_embedder_params.items()))

        else:
            # Warn that the original behavior is deprecated
            warnings.warn(DeprecationWarning(u"the token embedders for BasicTextFieldEmbedder should now "
                                             u"be specified as a dict under the 'token_embedders' key, "
                                             u"not as top-level key-value pairs"))

            token_embedders = {}
            keys = list(params.keys())
            for key in keys:
                embedder_params = params.pop(key)
                token_embedders[key] = TokenEmbedder.from_params(vocab=vocab, params=embedder_params)

        params.assert_empty(cls.__name__)
        return cls(token_embedders, embedder_to_indexer_map, allow_unmatched_keys)

BasicTextFieldEmbedder = TextFieldEmbedder.register(u"basic")(BasicTextFieldEmbedder)
