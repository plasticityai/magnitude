# pylint: disable=no-self-use,invalid-name


from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.token import Token
from allennlp.data.tokenizers.word_splitter import LettersDigitsWordSplitter
from allennlp.data.tokenizers.word_splitter import SimpleWordSplitter
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
try:
    from itertools import izip
except:
    izip = zip



class TestSimpleWordSplitter(AllenNlpTestCase):
    def setUp(self):
        super(TestSimpleWordSplitter, self).setUp()
        self.word_splitter = SimpleWordSplitter()

    def test_tokenize_handles_complex_punctuation(self):
        sentence = u"this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = [u"this", u"(", u"sentence", u")", u"has", u"'", u"crazy", u"'", u'"',
                           u"punctuation", u'"', u"."]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_contraction(self):
        sentence = u"it ain't joe's problem; would've been yesterday"
        expected_tokens = [u"it", u"ai", u"n't", u"joe", u"'s", u"problem", u";", u"would", u"'ve", u"been",
                           u"yesterday"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_batch_tokenization(self):
        sentences = [u"This is a sentence",
                     u"This isn't a sentence.",
                     u"This is the 3rd sentence."
                     u"Here's the 'fourth' sentence."]
        batch_split = self.word_splitter.batch_split_words(sentences)
        separately_split = [self.word_splitter.split_words(sentence) for sentence in sentences]
        assert len(batch_split) == len(separately_split)
        for batch_sentence, separate_sentence in izip(batch_split, separately_split):
            assert len(batch_sentence) == len(separate_sentence)
            for batch_word, separate_word in izip(batch_sentence, separate_sentence):
                assert batch_word.text == separate_word.text

    def test_tokenize_handles_multiple_contraction(self):
        sentence = u"wouldn't've"
        expected_tokens = [u"would", u"n't", u"'ve"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_final_apostrophe(self):
        sentence = u"the jones' house"
        expected_tokens = [u"the", u"jones", u"'", u"house"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_special_cases(self):
        sentence = u"mr. and mrs. jones, etc., went to, e.g., the store"
        expected_tokens = [u"mr.", u"and", u"mrs.", u"jones", u",", u"etc.", u",", u"went", u"to", u",",
                           u"e.g.", u",", u"the", u"store"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens


class TestLettersDigitsWordSplitter(AllenNlpTestCase):
    def setUp(self):
        super(TestLettersDigitsWordSplitter, self).setUp()
        self.word_splitter = LettersDigitsWordSplitter()

    def test_tokenize_handles_complex_punctuation(self):
        sentence = u"this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = [u"this", u"(", u"sentence", u")", u"has", u"'", u"crazy", u"'", u'"',
                           u"punctuation", u'"', u"."]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_unicode_letters(self):
        sentence = u"HAL9000   and    Ångström"
        expected_tokens = [Token(u"HAL", 0), Token(u"9000", 3), Token(u"and", 10), Token(u"Ångström", 17)]
        tokens = self.word_splitter.split_words(sentence)
        assert [t.text for t in tokens] == [t.text for t in expected_tokens]
        assert [t.idx for t in tokens] == [t.idx for t in expected_tokens]

    def test_tokenize_handles_splits_all_punctuation(self):
        sentence = u"wouldn't.[have] -3.45(m^2)"
        expected_tokens = [u"wouldn", u"'", u"t", u".", u"[", u"have", u"]", u"-", u"3",
                           u".", u"45", u"(", u"m", u"^", u"2", u")"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens


class TestSpacyWordSplitter(AllenNlpTestCase):
    def setUp(self):
        super(TestSpacyWordSplitter, self).setUp()
        self.word_splitter = SpacyWordSplitter()

    def test_tokenize_handles_complex_punctuation(self):
        sentence = u"this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = [u"this", u"(", u"sentence", u")", u"has", u"'", u"crazy", u"'", u'"',
                           u"punctuation", u'"', u"."]
        tokens = self.word_splitter.split_words(sentence)
        token_text = [t.text for t in tokens]
        assert token_text == expected_tokens
        for token in tokens:
            start = token.idx
            end = start + len(token.text)
            assert sentence[start:end] == token.text

    def test_tokenize_handles_contraction(self):
        # note that "would've" is kept together, while "ain't" is not.
        sentence = u"it ain't joe's problem; would been yesterday"
        expected_tokens = [u"it", u"ai", u"n't", u"joe", u"'s", u"problem", u";", u"would", u"been",
                           u"yesterday"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_multiple_contraction(self):
        sentence = u"wouldn't've"
        expected_tokens = [u"would", u"n't", u"'ve"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_final_apostrophe(self):
        sentence = u"the jones' house"
        expected_tokens = [u"the", u"jones", u"'", u"house"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_removes_whitespace_tokens(self):
        sentence = u"the\n jones'   house  \x0b  55"
        expected_tokens = [u"the", u"jones", u"'", u"house", u"55"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_tokenize_handles_special_cases(self):
        # note that the etc. doesn't quite work --- we can special case this if we want.
        sentence = u"Mr. and Mrs. Jones, etc., went to, e.g., the store"
        expected_tokens = [u"Mr.", u"and", u"Mrs.", u"Jones", u",", u"etc", u".", u",", u"went", u"to", u",",
                           u"e.g.", u",", u"the", u"store"]
        tokens = [t.text for t in self.word_splitter.split_words(sentence)]
        assert tokens == expected_tokens

    def test_batch_tokenization(self):
        sentences = [u"This is     a sentence",
                     u"This isn't a sentence.",
                     u"This is the 3rd     sentence."
                     u"Here's the 'fourth' sentence."]
        batch_split = self.word_splitter.batch_split_words(sentences)
        separately_split = [self.word_splitter.split_words(sentence) for sentence in sentences]
        assert len(batch_split) == len(separately_split)
        for batch_sentence, separate_sentence in izip(batch_split, separately_split):
            assert len(batch_sentence) == len(separate_sentence)
            for batch_word, separate_word in izip(batch_sentence, separate_sentence):
                assert batch_word.text == separate_word.text
