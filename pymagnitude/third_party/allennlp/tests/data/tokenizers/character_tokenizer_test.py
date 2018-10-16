# pylint: disable=no-self-use,invalid-name


from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import CharacterTokenizer
try:
    from itertools import izip
except:
    izip = zip


class TestCharacterTokenizer(AllenNlpTestCase):
    def test_splits_into_characters(self):
        tokenizer = CharacterTokenizer(start_tokens=[u'<S1>', u'<S2>'], end_tokens=[u'</S2>', u'</S1>'])
        sentence = u"A, small sentence."
        tokens = [t.text for t in tokenizer.tokenize(sentence)]
        expected_tokens = [u"<S1>", u"<S2>", u"A", u",", u" ", u"s", u"m", u"a", u"l", u"l", u" ", u"s", u"e",
                           u"n", u"t", u"e", u"n", u"c", u"e", u".", u'</S2>', u'</S1>']
        assert tokens == expected_tokens

    def test_batch_tokenization(self):
        tokenizer = CharacterTokenizer()
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

    def test_handles_byte_encoding(self):
        tokenizer = CharacterTokenizer(byte_encoding=u'utf-8', start_tokens=[259], end_tokens=[260])
        word = u"åøâáabe"
        tokens = [t.text_id for t in tokenizer.tokenize(word)]
        # Note that we've added one to the utf-8 encoded bytes, to account for masking.
        expected_tokens = [259, 196, 166, 196, 185, 196, 163, 196, 162, 98, 99, 102, 260]
        assert tokens == expected_tokens
