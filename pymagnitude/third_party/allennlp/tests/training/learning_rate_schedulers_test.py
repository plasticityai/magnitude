# pylint: disable=no-self-use,invalid-name,protected-access



from __future__ import with_statement
from __future__ import absolute_import
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.training.optimizers import Optimizer
from allennlp.common.testing import AllenNlpTestCase
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.common.params import Params


class LearningRateSchedulersTest(AllenNlpTestCase):

    def test_reduce_on_plateau_error_throw_when_no_metrics_exist(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        with self.assertRaises(ConfigurationError) as context:
            LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                    Params({u"type": u"adam"})),
                                              Params({u"type": u"reduce_on_plateau"})).step(None, None)

        self.assertTrue(
                u'The reduce_on_plateau learning rate scheduler requires a validation metric'
                in unicode(context.exception))

    def test_reduce_on_plateau_works_when_metrics_exist(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                Params({u"type": u"adam"})),
                                          Params({u"type": u"reduce_on_plateau"})).step(10, None)

    def test_no_metric_wrapper_can_support_none_for_metrics(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        lrs = LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                      Params({u"type": u"adam"})),
                                                Params({u"type": u"step", u"step_size": 1}))
        lrs.step(None, None)

    def test_noam_learning_rate_schedule_does_not_crash(self):
        model = torch.nn.Sequential(torch.nn.Linear(10, 10))
        lrs = LearningRateScheduler.from_params(Optimizer.from_params(model.named_parameters(),
                                                                      Params({u"type": u"adam"})),
                                                Params({u"type": u"noam", u"model_size": 10, u"warmup_steps": 2000}))
        lrs.step(None)
        lrs.step_batch(None)
