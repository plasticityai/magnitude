# pylint: disable=invalid-name,protected-access



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from flaky import flaky
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model


class CrfTaggerTest(ModelTestCase):
    def setUp(self):
        super(CrfTaggerTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / u'crf_tagger' / u'experiment.json',
                          self.FIXTURES_ROOT / u'data' / u'conll2003.txt')

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    @flaky
    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_forward_pass_runs_correctly(self):
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict[u'tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace=u"labels")
                assert tag in set([u'O', u'I-ORG', u'I-PER', u'I-LOC'])

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params[u"model"][u"encoder"][u"input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.pop(u"model"))
