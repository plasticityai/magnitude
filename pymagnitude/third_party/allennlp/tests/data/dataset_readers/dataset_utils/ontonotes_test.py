# pylint: disable=no-self-use,invalid-name



from __future__ import division
from __future__ import absolute_import
from nltk import Tree

from allennlp.data.dataset_readers.dataset_utils import Ontonotes
from allennlp.common.testing import AllenNlpTestCase


class TestOntonotes(AllenNlpTestCase):

    def test_dataset_iterator(self):
        reader = Ontonotes()
        annotated_sentences = list(reader.dataset_iterator(self.FIXTURES_ROOT / u'conll_2012' / u'subdomain'))
        annotation = annotated_sentences[0]
        assert annotation.document_id == u"test/test/01/test_001"
        assert annotation.sentence_id == 0
        assert annotation.words == [u'Mali', u'government', u'officials', u'say', u'the', u'woman',
                                    u"'s", u'confession', u'was', u'forced', u'.']
        assert annotation.pos_tags == [u'NNP', u'NN', u'NNS', u'VBP', u'DT',
                                       u'NN', u'POS', u'NN', u'VBD', u'JJ', u'.']
        assert annotation.word_senses == [None, None, 1, 1, None, 2, None, None, 1, None, None]
        assert annotation.predicate_framenet_ids == [None, None, None, u'01', None,
                                                     None, None, None, u'01', None, None]
        assert annotation.srl_frames == [(u"say", [u'B-ARG0', u'I-ARG0', u'I-ARG0', u'B-V', u'B-ARG1',
                                                  u'I-ARG1', u'I-ARG1', u'I-ARG1', u'I-ARG1', u'I-ARG1', u'O']),
                                         (u"was", [u'O', u'O', u'O', u'O', u'B-ARG1', u'I-ARG1', u'I-ARG1',
                                                  u'I-ARG1', u'B-V', u'B-ARG2', u'O'])]
        assert annotation.named_entities == [u'B-GPE', u'O', u'O', u'O', u'O', u'O',
                                             u'O', u'O', u'O', u'O', u'O']
        assert annotation.predicate_lemmas == [None, None, u'official', u'say', None,
                                               u'man', None, None, u'be', None, None]
        assert annotation.speakers == [None, None, None, None, None, None,
                                       None, None, None, None, None]

        assert annotation.parse_tree == Tree.fromstring(u"(TOP(S(NP(NML (NNP Mali)  (NN government) )"
                                                        u" (NNS officials) )(VP (VBP say) (SBAR(S(NP(NP"
                                                        u" (DT the)  (NN woman)  (POS 's) ) (NN "
                                                        u"confession) )(VP (VBD was) (ADJP (JJ "
                                                        u"forced) ))))) (. .) ))")
        assert annotation.coref_spans == set([(1, (4, 6)), (3, (4, 7))])

        annotation = annotated_sentences[1]
        assert annotation.document_id == u"test/test/02/test_002"
        assert annotation.sentence_id == 0
        assert annotation.words == [u'The', u'prosecution', u'rested', u'its', u'case', u'last', u'month',
                                    u'after', u'four', u'months', u'of', u'hearings', u'.']
        assert annotation.pos_tags == [u'DT', u'NN', u'VBD', u'PRP$', u'NN', u'JJ', u'NN',
                                       u'IN', u'CD', u'NNS', u'IN', u'NNS', u'.']
        assert annotation.word_senses == [None, 2, 5, None, 2, None, None,
                                          None, None, 1, None, 1, None]
        assert annotation.predicate_framenet_ids == [None, None, u'01', None, None, None,
                                                     None, None, None, None, None, u'01', None]
        assert annotation.srl_frames == [(u'rested', [u'B-ARG0', u'I-ARG0', u'B-V', u'B-ARG1',
                                                     u'I-ARG1', u'B-ARGM-TMP', u'I-ARGM-TMP',
                                                     u'B-ARGM-TMP', u'I-ARGM-TMP', u'I-ARGM-TMP',
                                                     u'I-ARGM-TMP', u'I-ARGM-TMP', u'O']),
                                         (u'hearings', [u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O',
                                                       u'O', u'O', u'O', u'B-V', u'O'])]
        assert annotation.named_entities == [u'O', u'O', u'O', u'O', u'O', u'B-DATE', u'I-DATE',
                                             u'O', u'B-DATE', u'I-DATE', u'O', u'O', u'O']
        assert annotation.predicate_lemmas == [None, u'prosecution', u'rest', None, u'case',
                                               None, None, None, None, u'month', None, u'hearing', None]
        assert annotation.speakers == [None, None, None, None, None, None,
                                       None, None, None, None, None, None, None]
        assert annotation.parse_tree == Tree.fromstring(u"(TOP(S(NP (DT The)  (NN prosecution) )(VP "
                                                        u"(VBD rested) (NP (PRP$ its)  (NN case) )"
                                                        u"(NP (JJ last)  (NN month) )(PP (IN after) "
                                                        u"(NP(NP (CD four)  (NNS months) )(PP (IN"
                                                        u" of) (NP (NNS hearings) ))))) (. .) ))")
        assert annotation.coref_spans == set([(2, (0, 1)), (2, (3, 3))])

        # Check we can handle sentences without verbs.
        annotation = annotated_sentences[2]
        assert annotation.document_id == u'test/test/03/test_003'
        assert annotation.sentence_id == 0
        assert annotation.words == [u'Denise', u'Dillon', u'Headline', u'News', u'.']
        assert annotation.pos_tags == [u'NNP', u'NNP', u'NNP', u'NNP', u'.']
        assert annotation.word_senses == [None, None, None, None, None]
        assert annotation.predicate_framenet_ids == [None, None, None, None, None]
        assert annotation.srl_frames == []
        assert annotation.named_entities == [u'B-PERSON', u'I-PERSON',
                                             u'B-WORK_OF_ART', u'I-WORK_OF_ART', u'O']
        assert annotation.predicate_lemmas == [None, None, None, None, None]
        assert annotation.speakers == [None, None, None, None, None]
        assert annotation.parse_tree == Tree.fromstring(u"(TOP(FRAG(NP (NNP Denise) "
                                                        u" (NNP Dillon) )(NP (NNP Headline)  "
                                                        u"(NNP News) ) (. .) ))")
        assert annotation.coref_spans == set([(2, (0, 1))])

        # Check we can handle sentences with 2 identical verbs.
        annotation = annotated_sentences[3]
        assert annotation.document_id == u'test/test/04/test_004'
        assert annotation.sentence_id == 0
        assert annotation.words == [u'and', u'that', u'wildness', u'is', u'still', u'in', u'him', u',',
                                    u'as', u'it', u'is', u'with', u'all', u'children', u'.']
        assert annotation.pos_tags == [u'CC', u'DT', u'NN', u'VBZ', u'RB', u'IN', u'PRP', u',',
                                       u'IN', u'PRP', u'VBZ', u'IN', u'DT', u'NNS', u'.']
        assert annotation.word_senses == [None, None, None, 4.0, None, None, None, None,
                                          None, None, 5.0, None, None, None, None]
        assert annotation.predicate_framenet_ids == [None, None, None, u'01', None, None,
                                                     None, None, None, None, u'01', None, None, None, None]
        assert annotation.srl_frames == [(u'is', [u'B-ARGM-DIS', u'B-ARG1', u'I-ARG1',
                                                 u'B-V', u'B-ARGM-TMP', u'B-ARG2', u'I-ARG2',
                                                 u'O', u'B-ARGM-ADV', u'I-ARGM-ADV', u'I-ARGM-ADV',
                                                 u'I-ARGM-ADV', u'I-ARGM-ADV', u'I-ARGM-ADV', u'O']),
                                         (u'is', [u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O',
                                                 u'B-ARG1', u'B-V', u'B-ARG2', u'I-ARG2', u'I-ARG2', u'O'])]
        assert annotation.named_entities == [u'O', u'O', u'O', u'O', u'O', u'O', u'O', u'O',
                                             u'O', u'O', u'O', u'O', u'O', u'O', u'O']
        assert annotation.predicate_lemmas == [None, None, None, u'be', None, None, None,
                                               None, None, None, u'be', None, None, None, None]
        assert annotation.speakers == [u'_Avalon_', u'_Avalon_', u'_Avalon_', u'_Avalon_', u'_Avalon_',
                                       u'_Avalon_', u'_Avalon_', u'_Avalon_', u'_Avalon_', u'_Avalon_',
                                       u'_Avalon_', u'_Avalon_', u'_Avalon_', u'_Avalon_', u'_Avalon_']
        assert annotation.parse_tree == Tree.fromstring(u"(TOP (S (CC and) (NP (DT that) (NN wildness)) "
                                                        u"(VP (VBZ is) (ADVP (RB still)) (PP (IN in) (NP "
                                                        u"(PRP him))) (, ,) (SBAR (IN as) (S (NP (PRP it)) "
                                                        u"(VP (VBZ is) (PP (IN with) (NP (DT all) (NNS "
                                                        u"children))))))) (. .)))")
        assert annotation.coref_spans == set([(14, (6, 6))])

    def test_dataset_path_iterator(self):
        reader = Ontonotes()
        files = list(reader.dataset_path_iterator(self.FIXTURES_ROOT / u'conll_2012'))
        expected_paths = [unicode(self.FIXTURES_ROOT / u'conll_2012' / u'subdomain' / u'example.gold_conll'),
                          unicode(self.FIXTURES_ROOT / u'conll_2012' / u'subdomain2' / u'example.gold_conll')]
        assert len(files) == len(expected_paths)
        assert set(files) == set(expected_paths)

    def test_ontonotes_can_read_conll_file_with_multiple_documents(self):
        reader = Ontonotes()
        file_path = self.FIXTURES_ROOT / u'coref' / u'coref.gold_conll'
        documents = list(reader.dataset_document_iterator(file_path))
        assert len(documents) == 2
