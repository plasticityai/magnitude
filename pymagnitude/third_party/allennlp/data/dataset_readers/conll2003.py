

from __future__ import with_statement
from __future__ import absolute_import
#typing
import itertools
import logging

#overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from io import open
try:
    from itertools import izip
except:
    izip = zip


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def _is_divider(line     )        :
    empty_line = line.strip() == u''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == u"-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False

_VALID_LABELS = set([u'ner', u'pos', u'chunk'])


class Conll2003DatasetReader(DatasetReader):
    u"""
    Reads instances from a pretokenised file where each line is in the following format:

    WORD POS-TAG CHUNK-TAG NER-TAG

    with a blank line indicating the end of each sentence
    and '-DOCSTART- -X- -X- O' indicating the end of each article,
    and converts it into a ``Dataset`` suitable for sequence tagging.

    Each ``Instance`` contains the words in the ``"tokens"`` ``TextField``.
    The values corresponding to the ``tag_label``
    values will get loaded into the ``"tags"`` ``SequenceLabelField``.
    And if you specify any ``feature_labels`` (you probably shouldn't),
    the corresponding values will get loaded into their own ``SequenceLabelField`` s.

    This dataset reader ignores the "article" divisions and simply treats
    each sentence as an independent ``Instance``. (Technically the reader splits sentences
    on any combination of blank lines and "DOCSTART" tags; in particular, it does the right
    thing on well formed inputs.)

    Parameters
    ----------
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
    tag_label: ``str``, optional (default=``ner``)
        Specify `ner`, `pos`, or `chunk` to have that tag loaded into the instance field `tag`.
    feature_labels: ``Sequence[str]``, optional (default=``()``)
        These labels will be loaded as features into the corresponding instance fields:
        ``pos`` -> ``pos_tags``, ``chunk`` -> ``chunk_tags``, ``ner`` -> ``ner_tags``
        Each will have its own namespace: ``pos_labels``, ``chunk_labels``, ``ner_labels``.
        If you want to use one of the labels as a `feature` in your model, it should be
        specified here.
    coding_scheme: ``str``, optional (default=``IOB1``)
        Specifies the coding scheme for ``ner_labels`` and ``chunk_labels``.
        Valid options are ``IOB1`` and ``BIOUL``.  The ``IOB1`` default maintains
        the original IOB1 scheme in the CoNLL data.
        In the IOB1 scheme, I is a token inside a span, O is a token outside
        a span and B is the beginning of span immediately following another
        span of the same type.
    """
    def __init__(self,
                 token_indexers                          = None,
                 tag_label      = u"ner",
                 feature_labels                = (),
                 lazy       = False,
                 coding_scheme      = u"IOB1")        :
        super(Conll2003DatasetReader, self).__init__(lazy)
        self._token_indexers = token_indexers or {u'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in _VALID_LABELS:
            raise ConfigurationError(u"unknown tag label type: {}".format(tag_label))
        for label in feature_labels:
            if label not in _VALID_LABELS:
                raise ConfigurationError(u"unknown feature label type: {}".format(label))
        if coding_scheme not in (u"IOB1", u"BIOUL"):
            raise ConfigurationError(u"unknown coding_scheme: {}".format(coding_scheme))

        self.tag_label = tag_label
        self.feature_labels = set(feature_labels)
        self.coding_scheme = coding_scheme

    #overrides
    def _read(self, file_path     )                      :
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, u"r") as data_file:
            logger.info(u"Reading instances from lines in file at: %s", file_path)

            # Group into alternative divider / sentence chunks.
            for is_divider, lines in itertools.groupby(data_file, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    tokens, pos_tags, chunk_tags, ner_tags = [list(field) for field in izip(*fields)]
                    # TextField requires ``Token`` objects
                    tokens = [Token(token) for token in tokens]

                    yield self.text_to_instance(tokens, pos_tags, chunk_tags, ner_tags)

    def text_to_instance(self, # type: ignore
                         tokens             ,
                         pos_tags            = None,
                         chunk_tags            = None,
                         ner_tags            = None)            :
        u"""
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        # pylint: disable=arguments-differ
        sequence = TextField(tokens, self._token_indexers)
        instance_fields                   = {u'tokens': sequence}
        instance_fields[u"metadata"] = MetadataField({u"words": [x.text for x in tokens]})

        # Recode the labels if necessary.
        if self.coding_scheme == u"BIOUL":
            coded_chunks = to_bioul(chunk_tags) if chunk_tags is not None else None
            coded_ner = to_bioul(ner_tags) if ner_tags is not None else None
        else:
            # the default IOB1
            coded_chunks = chunk_tags
            coded_ner = ner_tags

        # Add "feature labels" to instance
        if u'pos' in self.feature_labels:
            if pos_tags is None:
                raise ConfigurationError(u"Dataset reader was specified to use pos_tags as "
                                         u"features. Pass them to text_to_instance.")
            instance_fields[u'pos_tags'] = SequenceLabelField(pos_tags, sequence, u"pos_tags")
        if u'chunk' in self.feature_labels:
            if coded_chunks is None:
                raise ConfigurationError(u"Dataset reader was specified to use chunk tags as "
                                         u"features. Pass them to text_to_instance.")
            instance_fields[u'chunk_tags'] = SequenceLabelField(coded_chunks, sequence, u"chunk_tags")
        if u'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError(u"Dataset reader was specified to use NER tags as "
                                         u" features. Pass them to text_to_instance.")
            instance_fields[u'ner_tags'] = SequenceLabelField(coded_ner, sequence, u"ner_tags")

        # Add "tag label" to instance
        if self.tag_label == u'ner' and coded_ner is not None:
            instance_fields[u'tags'] = SequenceLabelField(coded_ner, sequence)
        elif self.tag_label == u'pos' and pos_tags is not None:
            instance_fields[u'tags'] = SequenceLabelField(pos_tags, sequence)
        elif self.tag_label == u'chunk' and coded_chunks is not None:
            instance_fields[u'tags'] = SequenceLabelField(coded_chunks, sequence)

        return Instance(instance_fields)

Conll2003DatasetReader = DatasetReader.register(u"conll2003")(Conll2003DatasetReader)
