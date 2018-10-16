

from __future__ import division
from __future__ import absolute_import
from allennlp.data.dataset_readers import CcgBankDatasetReader
from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase


class TestCcgBankReader(AllenNlpTestCase):

    def test_read_from_file(self):

        reader = CcgBankDatasetReader()
        instances = ensure_list(reader.read(self.FIXTURES_ROOT / u'data' / u'ccgbank.txt'))

        assert len(instances) == 2

        instance = instances[0]
        fields = instance.fields
        tokens = [token.text for token in fields[u'tokens'].tokens]
        assert tokens == [u'Pierre', u'Vinken', u',', u'61', u'years', u'old', u',', u'will', u'join', u'the', u'board',
                          u'as', u'a', u'nonexecutive', u'director', u'Nov.', u'29', u'.']

        ccg_categories = fields[u'ccg_categories'].labels
        assert ccg_categories == [u'N/N', u'N', u',', u'N/N', u'N', u'(S[adj]\\NP)\\NP', u',', u'(S[dcl]\\NP)/(S[b]\\NP)',
                                  u'(S[b]\\NP)/NP', u'NP[nb]/N', u'N', u'((S\\NP)\\(S\\NP))/NP', u'NP[nb]/N', u'N/N',
                                  u'N', u'((S\\NP)\\(S\\NP))/N[num]', u'N[num]', u'.']

        original_pos_tags = fields[u'original_pos_tags'].labels
        assert original_pos_tags == [u'NNP', u'NNP', u',', u'CD', u'NNS', u'JJ', u',', u'MD', u'VB', u'DT', u'NN',
                                     u'IN', u'DT', u'JJ', u'NN', u'NNP', u'CD', u'.']

        modified_pos_tags = fields[u'modified_pos_tags'].labels
        assert modified_pos_tags == [u'NNP', u'NNP', u',', u'CD', u'NNS', u'JJ', u',', u'MD', u'VB', u'DT', u'NN',
                                     u'IN', u'DT', u'JJ', u'NN', u'NNP', u'CD', u'.']

        predicate_arg_categories = fields[u'predicate_arg_categories'].labels
        assert predicate_arg_categories == [u'N_73/N_73', u'N', u',', u'N_93/N_93', u'N', u'(S[adj]\\NP_83)\\NP_84',
                                            u',', u'(S[dcl]\\NP_10)/(S[b]_11\\NP_10:B)_11', u'(S[b]\\NP)/NP',
                                            u'NP[nb]_29/N_29', u'N', u'((S_1\\NP_2)_1\\(S_1\\NP_2)_1)/NP',
                                            u'NP[nb]_48/N_48', u'N_43/N_43', u'N',
                                            u'((S_61\\NP_56)_61\\(S_61\\NP_56)_61)/N[num]_62', u'N[num]', u'.']
