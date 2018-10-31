

from __future__ import with_statement
from __future__ import absolute_import
#typing
import logging
import csv

#overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import JustSpacesWordSplitter
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from io import open

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class QuoraParaphraseDatasetReader(DatasetReader):
    u"""
    Reads a file from the Quora Paraphrase dataset. The train/validation/test split of the data
    comes from the paper `Bilateral Multi-Perspective Matching for Natural Language Sentences
    <https://arxiv.org/abs/1702.03814>`_ by Zhiguo Wang et al., 2017. Each file of the data
    is a tsv file without header. The columns are is_duplicate, question1, question2, and id.
    All questions are pre-tokenized and tokens are space separated. We convert these keys into
    fields named "label", "premise" and "hypothesis", so that it is compatible to some existing
    natural language inference algorithms.

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.  This also allows training with datasets that are too large to fit
        in memory.
    tokenizer : ``Tokenizer``, optional
        Tokenizer to use to split the premise and hypothesis into words or other kinds of tokens.
        Defaults to ``WordTokenizer(JustSpacesWordSplitter())``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        Indexers used to define input token representations. Defaults to ``{"tokens":
        SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 lazy       = False,
                 tokenizer            = None,
                 token_indexers                          = None)        :
        super(QuoraParaphraseDatasetReader, self).__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer(JustSpacesWordSplitter())
        self._token_indexers = token_indexers or {u"tokens": SingleIdTokenIndexer()}

    #overrides
    def _read(self, file_path):
        logger.info(u"Reading instances from lines in file at: %s", file_path)
        with open(cached_path(file_path), u"r") as data_file:
            tsv_in = csv.reader(data_file, delimiter=u'\t')
            for row in tsv_in:
                if len(row) == 4:
                    yield self.text_to_instance(premise=row[1], hypothesis=row[2], label=row[0])

    #overrides
    def text_to_instance(self,  # type: ignore
                         premise     ,
                         hypothesis     ,
                         label      = None)            :
        # pylint: disable=arguments-differ
        fields                   = {}
        tokenized_premise = self._tokenizer.tokenize(premise)
        tokenized_hypothesis = self._tokenizer.tokenize(hypothesis)
        fields[u"premise"] = TextField(tokenized_premise, self._token_indexers)
        fields[u"hypothesis"] = TextField(tokenized_hypothesis, self._token_indexers)
        if label is not None:
            fields[u'label'] = LabelField(label)

        return Instance(fields)

QuoraParaphraseDatasetReader = DatasetReader.register(u"quora_paraphrase")(QuoraParaphraseDatasetReader)
