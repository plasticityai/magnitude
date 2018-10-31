# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
#typing

import pytest

from allennlp.data.dataset_readers import WinobiasReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
try:
    from itertools import izip
except:
    izip = zip


class TestWinobiasReader(object):
    span_width = 5

    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = WinobiasReader(max_span_width=self.span_width, lazy=lazy)
        instances = ensure_list(conll_reader.read(unicode(AllenNlpTestCase.FIXTURES_ROOT /
                                                      u'coref' / u'winobias.sample')))

        assert len(instances) == 2

        fields = instances[0].fields
        text = [x.text for x in fields[u"text"].tokens]
        assert text == [u'The', u'designer', u'argued', u'with', u'the', u'developer',
                        u'and', u'slapped', u'her', u'in', u'the', u'face', u'.']

        spans = fields[u"spans"].field_list
        span_starts, span_ends = izip(*[(field.span_start, field.span_end) for field in spans])

        candidate_mentions = self.check_candidate_mentions_are_well_defined(span_starts, span_ends, text)

        gold_span_labels = fields[u"span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids                              = [(candidate_mentions[i], x)
                                                               for i, x in gold_indices_with_ids]
        assert gold_mentions_with_ids == [([u'the', u'developer'], 0), ([u'her'], 0)]

        fields = instances[1].fields
        text = [x.text for x in fields[u"text"].tokens]
        assert text == [u'The', u'salesperson', u'sold', u'some', u'books', u'to', u'the',
                        u'librarian', u'because', u'she', u'was', u'trying', u'to', u'sell', u'them', u'.']

        spans = fields[u"spans"].field_list
        span_starts, span_ends = izip(*[(field.span_start, field.span_end) for field in spans])
        candidate_mentions = self.check_candidate_mentions_are_well_defined(span_starts, span_ends, text)

        gold_span_labels = fields[u"span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids                              = [(candidate_mentions[i], x)
                                                               for i, x in gold_indices_with_ids]
        assert gold_mentions_with_ids == [([u'The', u'salesperson'], 0),
                                          ([u'some', u'books'], 1),
                                          ([u'she'], 0), ([u'them'], 1)]

    def check_candidate_mentions_are_well_defined(self, span_starts, span_ends, text):
        candidate_mentions = []
        for start, end in izip(span_starts, span_ends):
            # Spans are inclusive.
            text_span = text[start:(end + 1)]
            candidate_mentions.append(text_span)

        # Check we aren't considering zero length spans and all
        # candidate spans are less than what we specified
        assert all([self.span_width >= len(x) > 0 for x in candidate_mentions])  # pylint: disable=len-as-condition
        return candidate_mentions
