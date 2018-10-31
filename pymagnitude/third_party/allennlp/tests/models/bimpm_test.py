# pylint: disable=no-self-use,invalid-name



from __future__ import division
from __future__ import absolute_import
from allennlp.common.testing import ModelTestCase


class TestBiMPM(ModelTestCase):
    def setUp(self):
        super(TestBiMPM, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / u'bimpm' / u'experiment.json',
                          self.FIXTURES_ROOT / u'data' / u'quora_paraphrase.tsv')

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        assert u"logits" in output_dict and u"loss" in output_dict

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()
