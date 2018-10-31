
from __future__ import absolute_import
import logging
#typing

#overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.dataset_readers.dataset_utils import Ontonotes, OntonotesSentence


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SrlReader(DatasetReader):
    u"""
    This DatasetReader is designed to read in the English OntoNotes v5.0 data
    for semantic role labelling. It returns a dataset of instances with the
    following fields:

    tokens : ``TextField``
        The tokens in the sentence.
    verb_indicator : ``SequenceLabelField``
        A sequence of binary indicators for whether the word is the verb for this frame.
    tags : ``SequenceLabelField``
        A sequence of Propbank tags for the given verb in a BIO format.

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    domain_identifier: ``str``, (default = None)
        A string denoting a sub-domain of the Ontonotes 5.0 dataset to use. If present, only
        conll files under paths containing this domain identifier will be processed.

    Returns
    -------
    A ``Dataset`` of ``Instances`` for Semantic Role Labelling.

    """
    def __init__(self,
                 token_indexers                          = None,
                 domain_identifier      = None,
                 lazy       = False)        :
        super(SrlReader, self).__init__(lazy)
        self._token_indexers = token_indexers or {u"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier

    #overrides
    def _read(self, file_path     ):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        ontonotes_reader = Ontonotes()
        logger.info(u"Reading SRL instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info(u"Filtering to only include file paths containing the %s domain", self._domain_identifier)

        for sentence in self._ontonotes_subset(ontonotes_reader, file_path, self._domain_identifier):
            tokens = [Token(t) for t in sentence.words]
            if not sentence.srl_frames:
                # Sentence contains no predicates.
                tags = [u"O" for _ in tokens]
                verb_label = [0 for _ in tokens]
                yield self.text_to_instance(tokens, verb_label, tags)
            else:
                for (_, tags) in sentence.srl_frames:
                    verb_indicator = [1 if label[-2:] == u"-V" else 0 for label in tags]
                    yield self.text_to_instance(tokens, verb_indicator, tags)

    @staticmethod
    def _ontonotes_subset(ontonotes_reader           ,
                          file_path     ,
                          domain_identifier     )                               :
        u"""
        Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
        If the domain identifier is present, only examples which contain the domain
        identifier in the file path are yielded.
        """
        for conll_file in ontonotes_reader.dataset_path_iterator(file_path):
            if domain_identifier is None or "/{domain_identifier}/" in conll_file:
                yield ontonotes_reader.sentence_iterator(conll_file)

    def text_to_instance(self,  # type: ignore
                         tokens             ,
                         verb_label           ,
                         tags            = None)            :
        u"""
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        fields                   = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields[u'tokens'] = text_field
        fields[u'verb_indicator'] = SequenceLabelField(verb_label, text_field)
        if tags:
            fields[u'tags'] = SequenceLabelField(tags, text_field)

        if all([x == 0 for x in verb_label]):
            verb = None
        else:
            verb = tokens[verb_label.index(1)].text
        fields[u"metadata"] = MetadataField({u"words": [x.text for x in tokens],
                                            u"verb": verb})
        return Instance(fields)

SrlReader = DatasetReader.register(u"srl")(SrlReader)
