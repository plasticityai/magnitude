# pylint: disable=invalid-name,protected-access



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from copy import deepcopy
import pytest


from allennlp.common.testing import ModelTestCase
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.models import Model


class BiattentiveClassificationNetworkTest(ModelTestCase):
    def setUp(self):
        super(BiattentiveClassificationNetworkTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / u'biattentive_classification_network' / u'experiment.json',
                          self.FIXTURES_ROOT / u'data' / u'sst.txt')

    def test_maxout_bcn_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_feedforward_bcn_can_train_save_and_load(self):
        # pylint: disable=line-too-long
        self.ensure_model_can_train_save_and_load(self.FIXTURES_ROOT / u'biattentive_classification_network' / u'feedforward_experiment.json')

    def test_input_and_output_elmo_bcn_can_train_save_and_load(self):
        # pylint: disable=line-too-long
        self.ensure_model_can_train_save_and_load(self.FIXTURES_ROOT / u'biattentive_classification_network' / u'elmo_experiment.json')

    def test_output_only_elmo_bcn_can_train_save_and_load(self):
        # pylint: disable=line-too-long
        self.ensure_model_can_train_save_and_load(self.FIXTURES_ROOT / u'biattentive_classification_network' / u'output_only_elmo_experiment.json')

    def test_batch_predictions_are_consistent(self):
        self.ensure_batch_predictions_are_consistent()

    def test_mismatching_dimensions_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # Make the encoder wrong - it should be 2 to match
        # the embedding dimension from the text_field_embedder.
        params[u"model"][u"encoder"][u"input_size"] = 10
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get(u"model"))

    def test_no_elmo_but_set_flags_throws_configuration_error(self):
        params = Params.from_file(self.param_file)
        # There is no elmo specified in self.param_file, but set
        # use_input_elmo and use_integrator_output_elmo to True.
        # use_input_elmo set to True
        tmp_params = deepcopy(params)
        tmp_params[u"model"][u"use_input_elmo"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get(u"model"))

        # use_integrator_output_elmo set to True
        tmp_params = deepcopy(params)
        tmp_params[u"model"][u"use_input_elmo"] = False
        tmp_params[u"model"][u"use_integrator_output_elmo"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get(u"model"))

        # both use_input_elmo and use_integrator_output_elmo set to True
        tmp_params = deepcopy(params)
        tmp_params[u"model"][u"use_input_elmo"] = True
        tmp_params[u"model"][u"use_integrator_output_elmo"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get(u"model"))

    def test_elmo_but_no_set_flags_throws_configuration_error(self):
        # pylint: disable=line-too-long
        params = Params.from_file(self.FIXTURES_ROOT / u'biattentive_classification_network' / u'elmo_experiment.json')
        # Elmo is specified in the model, but set both flags to false.
        params[u"model"][u"use_input_elmo"] = False
        params[u"model"][u"use_integrator_output_elmo"] = False
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=params.get(u"model"))

    def test_elmo_num_repr_set_flags_mismatch_throws_configuration_error(self):
        # pylint: disable=line-too-long
        params = Params.from_file(self.FIXTURES_ROOT / u'biattentive_classification_network' / u'elmo_experiment.json')
        # Elmo is specified in the model, with num_output_representations=2. Set
        # only one flag to true.
        tmp_params = deepcopy(params)
        tmp_params[u"model"][u"use_input_elmo"] = False
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get(u"model"))

        tmp_params = deepcopy(params)
        tmp_params[u"model"][u"use_input_elmo"] = True
        tmp_params[u"model"][u"use_integrator_output_elmo"] = False
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get(u"model"))

        # set num_output_representations to 1, and set both flags to True.
        tmp_params = deepcopy(params)
        tmp_params[u"model"][u"elmo"][u"num_output_representations"] = 1
        tmp_params[u"model"][u"use_input_elmo"] = True
        tmp_params[u"model"][u"use_integrator_output_elmo"] = True
        with pytest.raises(ConfigurationError):
            Model.from_params(vocab=self.vocab, params=tmp_params.get(u"model"))

    def test_no_elmo_tokenizer_throws_configuration_error(self):
        with pytest.raises(ConfigurationError):
            # pylint: disable=line-too-long
            self.ensure_model_can_train_save_and_load(
                    self.FIXTURES_ROOT / u'biattentive_classification_network' / u'broken_experiments' / u'no_elmo_tokenizer_for_elmo.json')

    def test_elmo_in_text_field_embedder_throws_configuration_error(self):
        with pytest.raises(ConfigurationError):
            # pylint: disable=line-too-long
            self.ensure_model_can_train_save_and_load(
                    self.FIXTURES_ROOT / u'biattentive_classification_network' / u'broken_experiments' / u'elmo_in_text_field_embedder.json')
