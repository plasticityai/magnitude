# pylint: disable=no-self-use,invalid-name


from __future__ import division
from __future__ import absolute_import
#typing

import pytest

from allennlp.data.dataset_readers import ConllCorefReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
try:
    from itertools import izip
except:
    izip = zip


class TestCorefReader(object):
    span_width = 5

    @pytest.mark.parametrize(u"lazy", (True, False))
    def test_read_from_file(self, lazy):
        conll_reader = ConllCorefReader(max_span_width=self.span_width, lazy=lazy)
        instances = ensure_list(conll_reader.read(unicode(AllenNlpTestCase.FIXTURES_ROOT /
                                                      u'coref' / u'coref.gold_conll')))

        assert len(instances) == 2

        fields = instances[0].fields
        text = [x.text for x in fields[u"text"].tokens]

        assert text == [u'In', u'the', u'summer', u'of', u'2005', u',', u'a', u'picture', u'that',
                        u'people', u'have', u'long', u'been', u'looking', u'forward', u'to',
                        u'started', u'emerging', u'with', u'frequency', u'in', u'various', u'major',
                        u'Hong', u'Kong', u'media', u'.', u'With', u'their', u'unique', u'charm', u',',
                        u'these', u'well', u'-', u'known', u'cartoon', u'images', u'once', u'again',
                        u'caused', u'Hong', u'Kong', u'to', u'be', u'a', u'focus', u'of', u'worldwide',
                        u'attention', u'.', u'The', u'world', u"'s", u'fifth', u'Disney', u'park',
                        u'will', u'soon', u'open', u'to', u'the', u'public', u'here', u'.']

        spans = fields[u"spans"].field_list
        span_starts, span_ends = izip(*[(field.span_start, field.span_end) for field in spans])

        candidate_mentions = self.check_candidate_mentions_are_well_defined(span_starts, span_ends, text)

        gold_span_labels = fields[u"span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids                              = [(candidate_mentions[i], x)
                                                               for i, x in gold_indices_with_ids]

        assert ([u"Hong", u"Kong"], 0) in gold_mentions_with_ids
        gold_mentions_with_ids.remove(([u"Hong", u"Kong"], 0))
        assert ([u"Hong", u"Kong"], 0) in gold_mentions_with_ids
        assert ([u"their"], 1) in gold_mentions_with_ids
        # This is a span which exceeds our max_span_width, so it should not be considered.
        assert not ([u"these", u"well", u"known", u"cartoon", u"images"], 1) in gold_mentions_with_ids

        fields = instances[1].fields
        text = [x.text for x in fields[u"text"].tokens]
        assert text == [u'The', u'area', u'of', u'Hong', u'Kong', u'is', u'only', u'one', u'thousand', u'-', u'plus',
                        u'square', u'kilometers', u'.', u'The', u'population', u'is', u'dense', u'.', u'Natural',
                        u'resources', u'are', u'relatively', u'scarce', u'.', u'However', u',', u'the', u'clever',
                        u'Hong', u'Kong', u'people', u'will', u'utilize', u'all', u'resources', u'they', u'have',
                        u'created', u'for', u'developing', u'the', u'Hong', u'Kong', u'tourism', u'industry', u'.']

        spans = fields[u"spans"].field_list
        span_starts, span_ends = izip(*[(field.span_start, field.span_end) for field in spans])

        candidate_mentions = self.check_candidate_mentions_are_well_defined(span_starts, span_ends, text)

        gold_span_labels = fields[u"span_labels"]
        gold_indices_with_ids = [(i, x) for i, x in enumerate(gold_span_labels.labels) if x != -1]
        gold_mentions_with_ids                              = [(candidate_mentions[i], x)
                                                               for i, x in gold_indices_with_ids]

        assert ([u"Hong", u"Kong"], 0) in gold_mentions_with_ids
        gold_mentions_with_ids.remove(([u"Hong", u"Kong"], 0))
        assert ([u"Hong", u"Kong"], 0) in gold_mentions_with_ids
        assert ([u"they"], 1) in gold_mentions_with_ids
        assert ([u'the', u'clever', u'Hong', u'Kong', u'people'], 1) in gold_mentions_with_ids

    def check_candidate_mentions_are_well_defined(self, span_starts, span_ends, text):
        candidate_mentions = []
        for start, end in izip(span_starts, span_ends):
            # Spans are inclusive.
            text_span = text[start: end + 1]
            candidate_mentions.append(text_span)

        # Check we aren't considering zero length spans and all
        # candidate spans are less than what we specified
        assert all([self.span_width >= len(x) > 0 for x in candidate_mentions])  # pylint: disable=len-as-condition
        return candidate_mentions
