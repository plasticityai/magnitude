# pylint: disable=no-self-use,invalid-name,protected-access


from __future__ import with_statement
from __future__ import absolute_import
#typing

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils import span_utils
from allennlp.data.tokenizers.word_tokenizer import SpacyWordSplitter
from allennlp.data.tokenizers.token import Token

class SpanUtilsTest(AllenNlpTestCase):

    def test_bio_tags_to_spans_extracts_correct_spans(self):
        tag_sequence = [u"O", u"B-ARG1", u"I-ARG1", u"O", u"B-ARG2", u"I-ARG2", u"B-ARG1", u"B-ARG2"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == set([(u"ARG1", (1, 2)), (u"ARG2", (4, 5)), (u"ARG1", (6, 6)), (u"ARG2", (7, 7))])

        # Check that it raises when we use U- tags for single tokens.
        tag_sequence = [u"O", u"B-ARG1", u"I-ARG1", u"O", u"B-ARG2", u"I-ARG2", u"U-ARG1", u"U-ARG2"]
        with self.assertRaises(span_utils.InvalidTagSequence):
            spans = span_utils.bio_tags_to_spans(tag_sequence)

        # Check that invalid BIO sequences are also handled as spans.
        tag_sequence = [u"O", u"B-ARG1", u"I-ARG1", u"O", u"I-ARG1", u"B-ARG2", u"I-ARG2", u"B-ARG1", u"I-ARG2", u"I-ARG2"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == set([(u"ARG1", (1, 2)), (u"ARG2", (5, 6)), (u"ARG1", (7, 7)),
                              (u"ARG1", (4, 4)), (u"ARG2", (8, 9))])

    def test_bio_tags_to_spans_extracts_correct_spans_without_labels(self):
        tag_sequence = [u"O", u"B", u"I", u"O", u"B", u"I", u"B", u"B"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == set([(u"", (1, 2)), (u"", (4, 5)), (u"", (6, 6)), (u"", (7, 7))])

        # Check that it raises when we use U- tags for single tokens.
        tag_sequence = [u"O", u"B", u"I", u"O", u"B", u"I", u"U", u"U"]
        with self.assertRaises(span_utils.InvalidTagSequence):
            spans = span_utils.bio_tags_to_spans(tag_sequence)

        # Check that invalid BIO sequences are also handled as spans.
        tag_sequence = [u"O", u"B", u"I", u"O", u"I", u"B", u"I", u"B", u"I", u"I"]
        spans = span_utils.bio_tags_to_spans(tag_sequence)
        assert set(spans) == set([(u'', (1, 2)), (u'', (4, 4)), (u'', (5, 6)), (u'', (7, 9))])

    def test_bio_tags_to_spans_ignores_specified_tags(self):
        tag_sequence = [u"B-V", u"I-V", u"O", u"B-ARG1", u"I-ARG1",
                        u"O", u"B-ARG2", u"I-ARG2", u"B-ARG1", u"B-ARG2"]
        spans = span_utils.bio_tags_to_spans(tag_sequence, [u"ARG1", u"V"])
        assert set(spans) == set([(u"ARG2", (6, 7)), (u"ARG2", (9, 9))])

    def test_iob1_tags_to_spans_extracts_correct_spans_without_labels(self):
        tag_sequence = [u"I", u"B", u"I", u"O", u"B", u"I", u"B", u"B"]
        spans = span_utils.iob1_tags_to_spans(tag_sequence)
        assert set(spans) == set([(u"", (0, 0)), (u"", (1, 2)), (u"", (4, 5)), (u"", (6, 6)), (u"", (7, 7))])

        # Check that it raises when we use U- tags for single tokens.
        tag_sequence = [u"O", u"B", u"I", u"O", u"B", u"I", u"U", u"U"]
        with self.assertRaises(span_utils.InvalidTagSequence):
            spans = span_utils.iob1_tags_to_spans(tag_sequence)

        # Check that invalid IOB1 sequences are also handled as spans.
        tag_sequence = [u"O", u"B", u"I", u"O", u"I", u"B", u"I", u"B", u"I", u"I"]
        spans = span_utils.iob1_tags_to_spans(tag_sequence)
        assert set(spans) == set([(u'', (1, 2)), (u'', (4, 4)), (u'', (5, 6)), (u'', (7, 9))])

    def test_iob1_tags_to_spans_extracts_correct_spans(self):
        tag_sequence = [u"I-ARG2", u"B-ARG1", u"I-ARG1", u"O", u"B-ARG2", u"I-ARG2", u"B-ARG1", u"B-ARG2"]
        spans = span_utils.iob1_tags_to_spans(tag_sequence)
        assert set(spans) == set([(u"ARG2", (0, 0)), (u"ARG1", (1, 2)), (u"ARG2", (4, 5)),
                              (u"ARG1", (6, 6)), (u"ARG2", (7, 7))])

        # Check that it raises when we use U- tags for single tokens.
        tag_sequence = [u"O", u"B-ARG1", u"I-ARG1", u"O", u"B-ARG2", u"I-ARG2", u"U-ARG1", u"U-ARG2"]
        with self.assertRaises(span_utils.InvalidTagSequence):
            spans = span_utils.iob1_tags_to_spans(tag_sequence)

        # Check that invalid IOB1 sequences are also handled as spans.
        tag_sequence = [u"O", u"B-ARG1", u"I-ARG1", u"O", u"I-ARG1", u"B-ARG2",
                        u"I-ARG2", u"B-ARG1", u"I-ARG2", u"I-ARG2"]
        spans = span_utils.iob1_tags_to_spans(tag_sequence)
        assert set(spans) == set([(u"ARG1", (1, 2)), (u"ARG1", (4, 4)), (u"ARG2", (5, 6)),
                              (u"ARG1", (7, 7)), (u"ARG2", (8, 9))])

    def test_enumerate_spans_enumerates_all_spans(self):
        tokenizer = SpacyWordSplitter(pos_tags=True)
        sentence = tokenizer.split_words(u"This is a sentence.")

        spans = span_utils.enumerate_spans(sentence)
        assert spans == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2),
                         (1, 3), (1, 4), (2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (4, 4)]

        spans = span_utils.enumerate_spans(sentence, max_span_width=3, min_span_width=2)
        assert spans == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]

        spans = span_utils.enumerate_spans(sentence, max_span_width=3, min_span_width=2, offset=20)
        assert spans == [(20, 21), (20, 22), (21, 22), (21, 23), (22, 23), (22, 24), (23, 24)]

        def no_prefixed_punctuation(tokens             ):
            # Only include spans which don't start or end with punctuation.
            return tokens[0].pos_ != u"PUNCT" and tokens[-1].pos_ != u"PUNCT"

        spans = span_utils.enumerate_spans(sentence,
                                           max_span_width=3,
                                           min_span_width=2,
                                           filter_function=no_prefixed_punctuation)

        # No longer includes (2, 4) or (3, 4) as these include punctuation
        # as their last element.
        assert spans == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]

    def test_bioul_tags_to_spans(self):
        tag_sequence = [u'B-PER', u'I-PER', u'L-PER', u'U-PER', u'U-LOC', u'O']
        spans = span_utils.bioul_tags_to_spans(tag_sequence)
        assert spans == [(u'PER', (0, 2)), (u'PER', (3, 3)), (u'LOC', (4, 4))]

        tag_sequence = [u'B-PER', u'I-PER', u'O']
        with self.assertRaises(span_utils.InvalidTagSequence):
            spans = span_utils.bioul_tags_to_spans(tag_sequence)

    def test_bioul_tags_to_spans_without_labels(self):
        tag_sequence = [u'B', u'I', u'L', u'U', u'U', u'O']
        spans = span_utils.bioul_tags_to_spans(tag_sequence)
        assert spans == [(u'', (0, 2)), (u'', (3, 3)), (u'', (4, 4))]

        tag_sequence = [u'B', u'I', u'O']
        with self.assertRaises(span_utils.InvalidTagSequence):
            spans = span_utils.bioul_tags_to_spans(tag_sequence)

    def test_iob1_to_bioul(self):
        tag_sequence = [u'I-ORG', u'O', u'I-MISC', u'O']
        bioul_sequence = span_utils.to_bioul(tag_sequence, encoding=u"IOB1")
        assert bioul_sequence == [u'U-ORG', u'O', u'U-MISC', u'O']

        tag_sequence = [u'O', u'I-PER', u'B-PER', u'I-PER', u'I-PER', u'B-PER']
        bioul_sequence = span_utils.to_bioul(tag_sequence, encoding=u"IOB1")
        assert bioul_sequence == [u'O', u'U-PER', u'B-PER', u'I-PER', u'L-PER', u'U-PER']

    def test_bio_to_bioul(self):
        tag_sequence = [u'B-ORG', u'O', u'B-MISC', u'O', u'B-MISC', u'I-MISC', u'I-MISC']
        bioul_sequence = span_utils.to_bioul(tag_sequence, encoding=u"BIO")
        assert bioul_sequence == [u'U-ORG', u'O', u'U-MISC', u'O', u'B-MISC', u'I-MISC', u'L-MISC']

        # Encoding in IOB format should throw error with incorrect encoding.
        with self.assertRaises(span_utils.InvalidTagSequence):
            tag_sequence = [u'O', u'I-PER', u'B-PER', u'I-PER', u'I-PER', u'B-PER']
            bioul_sequence = span_utils.to_bioul(tag_sequence, encoding=u"BIO")
