

from __future__ import with_statement
from __future__ import absolute_import
import json
import logging
#typing

#overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from io import open
try:
    from itertools import izip
except:
    izip = zip


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SquadReader(DatasetReader):
    u"""
    Reads a JSON-formatted SQuAD file and returns a ``Dataset`` where the ``Instances`` have four
    fields: ``question``, a ``TextField``, ``passage``, another ``TextField``, and ``span_start``
    and ``span_end``, both ``IndexFields`` into the ``passage`` ``TextField``.  We also add a
    ``MetadataField`` that stores the instance's ID, the original passage text, gold answer strings,
    and token offsets into the original passage, accessible as ``metadata['id']``,
    ``metadata['original_passage']``, ``metadata['answer_texts']`` and
    ``metadata['token_offsets']``.  This is so that we can more easily use the official SQuAD
    evaluation script to get metrics.

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the question and the passage.  See :class:`Tokenizer`.
        Default is ```WordTokenizer()``.
    token_indexers : ``Dict[str, TokenIndexer]``, optional
        We similarly use this for both the question and the passage.  See :class:`TokenIndexer`.
        Default is ``{"tokens": SingleIdTokenIndexer()}``.
    """
    def __init__(self,
                 tokenizer            = None,
                 token_indexers                          = None,
                 lazy       = False)        :
        super(SquadReader, self).__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {u'tokens': SingleIdTokenIndexer()}

    #overrides
    def _read(self, file_path     ):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info(u"Reading file at %s", file_path)
        with open(file_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json[u'data']
        logger.info(u"Reading the dataset")
        for article in dataset:
            for paragraph_json in article[u'paragraphs']:
                paragraph = paragraph_json[u"context"]
                tokenized_paragraph = self._tokenizer.tokenize(paragraph)

                for question_answer in paragraph_json[u'qas']:
                    question_text = question_answer[u"question"].strip().replace(u"\n", u"")
                    answer_texts = [answer[u'text'] for answer in question_answer[u'answers']]
                    span_starts = [answer[u'answer_start'] for answer in question_answer[u'answers']]
                    span_ends = [start + len(answer) for start, answer in izip(span_starts, answer_texts)]
                    instance = self.text_to_instance(question_text,
                                                     paragraph,
                                                     izip(span_starts, span_ends),
                                                     answer_texts,
                                                     tokenized_paragraph)
                    yield instance

    #overrides
    def text_to_instance(self,  # type: ignore
                         question_text     ,
                         passage_text     ,
                         char_spans                        = None,
                         answer_texts            = None,
                         passage_tokens              = None)            :
        # pylint: disable=arguments-differ
        if not passage_tokens:
            passage_tokens = self._tokenizer.tokenize(passage_text)
        char_spans = char_spans or []

        # We need to convert character indices in `passage_text` to token indices in
        # `passage_tokens`, as the latter is what we'll actually use for supervision.
        token_spans                        = []
        passage_offsets = [(token.idx, token.idx + len(token.text)) for token in passage_tokens]
        for char_span_start, char_span_end in char_spans:
            (span_start, span_end), error = util.char_span_to_token_span(passage_offsets,
                                                                         (char_span_start, char_span_end))
            if error:
                logger.debug(u"Passage: %s", passage_text)
                logger.debug(u"Passage tokens: %s", passage_tokens)
                logger.debug(u"Question text: %s", question_text)
                logger.debug(u"Answer span: (%d, %d)", char_span_start, char_span_end)
                logger.debug(u"Token span: (%d, %d)", span_start, span_end)
                logger.debug(u"Tokens in answer: %s", passage_tokens[span_start:span_end + 1])
                logger.debug(u"Answer: %s", passage_text[char_span_start:char_span_end])
            token_spans.append((span_start, span_end))

        return util.make_reading_comprehension_instance(self._tokenizer.tokenize(question_text),
                                                        passage_tokens,
                                                        self._token_indexers,
                                                        passage_text,
                                                        token_spans,
                                                        answer_texts)

SquadReader = DatasetReader.register(u"squad")(SquadReader)
