

from __future__ import with_statement
from __future__ import absolute_import
#typing
import json
import logging

#overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from io import open

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SnliReader(DatasetReader):
    u"""
    Reads a file from the Stanford Natural Language Inference (SNLI) dataset.  This data is
    formatted as jsonl, one json-formatted instance per line.  The keys in the data are
    "gold_label", "sentence1", and "sentence2".  We convert these keys into fields named "label",
    "premise" and "hypothesis", along with a metadata field containing the tokenized strings of the
    premise and hypothesis.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 tokenizer            = None,
                 token_indexers                          = None,
                 lazy       = False)        :
        super(SnliReader, self).__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {u'tokens': SingleIdTokenIndexer()}

    #overrides
    def _read(self, file_path     ):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, u'r') as snli_file:
            logger.info(u"Reading SNLI instances from jsonl dataset at: %s", file_path)
            for line in snli_file:
                example = json.loads(line)

                label = example[u"gold_label"]
                if label == u'-':
                    # These were cases where the annotators disagreed; we'll just skip them.  It's
                    # like 800 out of 500k examples in the training data.
                    continue

                premise = example[u"sentence1"]
                hypothesis = example[u"sentence2"]

                yield self.text_to_instance(premise, hypothesis, label)

    #overrides
    def text_to_instance(self,  # type: ignore
                         premise     ,
                         hypothesis     ,
                         label      = None)            :
        # pylint: disable=arguments-differ
        fields                   = {}
        premise_tokens = self._tokenizer.tokenize(premise)
        hypothesis_tokens = self._tokenizer.tokenize(hypothesis)
        fields[u'premise'] = TextField(premise_tokens, self._token_indexers)
        fields[u'hypothesis'] = TextField(hypothesis_tokens, self._token_indexers)
        if label:
            fields[u'label'] = LabelField(label)

        metadata = {u"premise_tokens": [x.text for x in premise_tokens],
                    u"hypothesis_tokens": [x.text for x in hypothesis_tokens]}
        fields[u"metadata"] = MetadataField(metadata)
        return Instance(fields)

SnliReader = DatasetReader.register(u"snli")(SnliReader)
