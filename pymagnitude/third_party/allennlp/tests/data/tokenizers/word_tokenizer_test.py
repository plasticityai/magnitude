# pylint: disable=no-self-use,invalid-name


from __future__ import absolute_import
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer
try:
    from itertools import izip
except:
    izip = zip


class TestWordTokenizer(AllenNlpTestCase):
    def test_passes_through_correctly(self):
        tokenizer = WordTokenizer(start_tokens=[u'@@', u'%%'], end_tokens=[u'^^'])
        sentence = u"this (sentence) has 'crazy' \"punctuation\"."
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        expected_tokens = [u"@@", u"%%", u"this", u"(", u"sentence", u")", u"has", u"'", u"crazy", u"'", u"\"",
                           u"punctuation", u"\"", u".", u"^^"]
        assert tokens == expected_tokens

    def test_batch_tokenization(self):
        tokenizer = WordTokenizer()
        sentences = [u"This is a sentence",
                     u"This isn't a sentence.",
                     u"This is the 3rd sentence."
                     u"Here's the 'fourth' sentence."]
        batch_tokenized = tokenizer.batch_tokenize(sentences)
        separately_tokenized = [tokenizer.tokenize(sentence) for sentence in sentences]
        assert len(batch_tokenized) == len(separately_tokenized)
        for batch_sentence, separate_sentence in izip(batch_tokenized, separately_tokenized):
            assert len(batch_sentence) == len(separate_sentence)
            for batch_word, separate_word in izip(batch_sentence, separate_sentence):
                assert batch_word.text == separate_word.text

    def test_stems_and_filters_correctly(self):
        tokenizer = WordTokenizer.from_params(Params({u'word_stemmer': {u'type': u'porter'},
                                                      u'word_filter': {u'type': u'stopwords'}}))
        sentence = u"this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = [u"sentenc", u"ha", u"crazi", u"punctuat"]
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        assert tokens == expected_tokens
