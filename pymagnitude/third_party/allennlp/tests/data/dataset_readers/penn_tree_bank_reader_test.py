# pylint: disable=no-self-use,invalid-name,protected-access



from __future__ import division
from __future__ import absolute_import
from nltk.tree import Tree

from allennlp.data.dataset_readers import PennTreeBankConstituencySpanDatasetReader
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers.dataset_utils.span_utils import enumerate_spans
try:
    from itertools import izip
except:
    izip = zip



class TestPennTreeBankConstituencySpanReader(AllenNlpTestCase):

    def setUp(self):
        super(TestPennTreeBankConstituencySpanReader, self).setUp()
        self.span_width = 5

    def test_read_from_file(self):

        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        instances = ptb_reader.read(unicode(self.FIXTURES_ROOT / u'data' / u'example_ptb.trees'))

        assert len(instances) == 2

        fields = instances[0].fields
        tokens = [x.text for x in fields[u"tokens"].tokens]
        pos_tags = fields[u"pos_tags"].labels
        spans = [(x.span_start, x.span_end) for x in fields[u"spans"].field_list]
        span_labels = fields[u"span_labels"].labels

        assert tokens == [u'Also', u',', u'because', u'UAL', u'Chairman', u'Stephen', u'Wolf',
                          u'and', u'other', u'UAL', u'executives', u'have', u'joined', u'the',
                          u'pilots', u"'", u'bid', u',', u'the', u'board', u'might', u'be', u'forced',
                          u'to', u'exclude', u'him', u'from', u'its', u'deliberations', u'in',
                          u'order', u'to', u'be', u'fair', u'to', u'other', u'bidders', u'.']
        assert pos_tags == [u'RB', u',', u'IN', u'NNP', u'NNP', u'NNP', u'NNP', u'CC', u'JJ', u'NNP',
                            u'NNS', u'VBP', u'VBN', u'DT', u'NNS', u'POS', u'NN', u',', u'DT', u'NN',
                            u'MD', u'VB', u'VBN', u'TO', u'VB', u'PRP', u'IN', u'PRP$',
                            u'NNS', u'IN', u'NN', u'TO', u'VB', u'JJ', u'TO', u'JJ', u'NNS', u'.']

        assert spans == enumerate_spans(tokens)
        gold_tree = Tree.fromstring(u"(S(ADVP(RB Also))(, ,)(SBAR(IN because)"
                                    u"(S(NP(NP(NNP UAL)(NNP Chairman)(NNP Stephen)(NNP Wolf))"
                                    u"(CC and)(NP(JJ other)(NNP UAL)(NNS executives)))(VP(VBP have)"
                                    u"(VP(VBN joined)(NP(NP(DT the)(NNS pilots)(POS '))(NN bid))))))"
                                    u"(, ,)(NP(DT the)(NN board))(VP(MD might)(VP(VB be)(VP(VBN "
                                    u"forced)(S(VP(TO to)(VP(VB exclude)(NP(PRP him))(PP(IN from)"
                                    u"(NP(PRP$ its)(NNS deliberations)))(SBAR(IN in)(NN order)(S("
                                    u"VP(TO to)(VP(VB be)(ADJP(JJ fair)(PP(TO to)(NP(JJ other)(NNS "
                                    u"bidders))))))))))))))(. .))")

        assert fields[u"metadata"].metadata[u"gold_tree"] == gold_tree
        assert fields[u"metadata"].metadata[u"tokens"] == tokens

        correct_spans_and_labels = {}
        ptb_reader._get_gold_spans(gold_tree, 0, correct_spans_and_labels)
        for span, label in izip(spans, span_labels):
            if label != u"NO-LABEL":
                assert correct_spans_and_labels[span] == label


        fields = instances[1].fields
        tokens = [x.text for x in fields[u"tokens"].tokens]
        pos_tags = fields[u"pos_tags"].labels
        spans = [(x.span_start, x.span_end) for x in fields[u"spans"].field_list]
        span_labels = fields[u"span_labels"].labels

        assert tokens == [u'That', u'could', u'cost', u'him', u'the', u'chance',
                          u'to', u'influence', u'the', u'outcome', u'and', u'perhaps',
                          u'join', u'the', u'winning', u'bidder', u'.']

        assert pos_tags == [u'DT', u'MD', u'VB', u'PRP', u'DT', u'NN',
                            u'TO', u'VB', u'DT', u'NN', u'CC', u'RB', u'VB', u'DT',
                            u'VBG', u'NN', u'.']

        assert spans == enumerate_spans(tokens)

        gold_tree = Tree.fromstring(u"(S(NP(DT That))(VP(MD could)(VP(VB cost)(NP(PRP him))"
                                    u"(NP(DT the)(NN chance)(S(VP(TO to)(VP(VP(VB influence)(NP(DT the)"
                                    u"(NN outcome)))(CC and)(VP(ADVP(RB perhaps))(VB join)(NP(DT the)"
                                    u"(VBG winning)(NN bidder)))))))))(. .))")

        assert fields[u"metadata"].metadata[u"gold_tree"] == gold_tree
        assert fields[u"metadata"].metadata[u"tokens"] == tokens

        correct_spans_and_labels = {}
        ptb_reader._get_gold_spans(gold_tree, 0, correct_spans_and_labels)
        for span, label in izip(spans, span_labels):
            if label != u"NO-LABEL":
                assert correct_spans_and_labels[span] == label

    def test_strip_functional_tags(self):
        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        # Get gold spans should strip off all the functional tags.
        tree = Tree.fromstring(u"(S (NP=PRP (D the) (N dog)) (VP-0 (V chased) (NP|FUN-TAGS (D the) (N cat))))")
        ptb_reader._strip_functional_tags(tree)
        assert tree == Tree.fromstring(u"(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")

    def test_get_gold_spans_correctly_extracts_spans(self):
        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        tree = Tree.fromstring(u"(S (NP (D the) (N dog)) (VP (V chased) (NP (D the) (N cat))))")

        span_dict = {}
        ptb_reader._get_gold_spans(tree, 0, span_dict)
        spans = list(span_dict.items()) # pylint: disable=protected-access
        assert spans == [((0, 1), u'NP'), ((3, 4), u'NP'), ((2, 4), u'VP'), ((0, 4), u'S')]

    def test_get_gold_spans_correctly_extracts_spans_with_nested_labels(self):
        ptb_reader = PennTreeBankConstituencySpanDatasetReader()
        # Here we have a parse with several nested labels - particularly the (WHNP (WHNP (WP What)))
        # fragment. These should be concatenated into a single label by get_gold_spans.
        tree = Tree.fromstring(u"""
            (S
        (`` ``)
        (S-TPC
        (NP-SBJ (PRP We))
        (VP
            (VBP have)
            (S
            (VP
                (TO to)
                (VP
                (VP
                    (VB clear)
                    (PRT (RP up))
                    (NP (DT these) (NNS issues)))
                (CC and)
                (VP
                    (VB find)
                    (PRT (RP out))
                    (SBAR-NOM
                    (WHNP (WHNP (WP what)))
                    (S
                        (VP
                        (VBZ is)
                        (ADJP-PRD (JJ present))
                        (SBAR
                            (WHNP (WDT that))
                            (S
                            (VP
                                (VBZ is)
                                (VP
                                (VBG creating)
                                (NP (JJ artificial) (NN volatility)))))))))))))))
        (, ,)
        ('' '')
        (NP-SBJ (NNP Mr.) (NNP Fisher))
        (VP (VBD said))
        (. .))
        """)
        span_dict = {}
        ptb_reader._strip_functional_tags(tree) # pylint: disable=protected-access
        ptb_reader._get_gold_spans(tree, 0, span_dict) # pylint: disable=protected-access
        assert span_dict == {(1, 1): u'NP', (5, 5): u'PRT', (6, 7): u'NP', (4, 7): u'VP', (10, 10): u'PRT',
                             (11, 11): u'WHNP-WHNP', (13, 13): u'ADJP', (14, 14): u'WHNP', (17, 18): u'NP',
                             (16, 18): u'VP', (15, 18): u'S-VP', (14, 18): u'SBAR', (12, 18): u'S-VP',
                             (11, 18): u'SBAR', (9, 18): u'VP', (4, 18): u'VP', (3, 18): u'S-VP',
                             (2, 18): u'VP', (1, 18): u'S', (21, 22): u'NP', (23, 23): u'VP', (0, 24): u'S'}
