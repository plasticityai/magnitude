# pylint: disable=no-self-use,invalid-name

from __future__ import absolute_import
import torch
from torch.nn.init import constant_
from allennlp.common.params import Params
from allennlp.nn import InitializerApplicator
from allennlp.nn.regularizers import L1Regularizer, L2Regularizer, RegularizerApplicator
from allennlp.common.testing import AllenNlpTestCase


class TestRegularizers(AllenNlpTestCase):
    def test_l1_regularization(self):
        model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.Linear(10, 5)
        )
        initializer = InitializerApplicator([(u".*", lambda tensor: constant_(tensor, -1))])
        initializer(model)
        value = RegularizerApplicator([(u"", L1Regularizer(1.0))])(model)
        # 115 because of biases.
        assert value.data.numpy() == 115.0

    def test_l2_regularization(self):
        model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.Linear(10, 5)
        )
        initializer = InitializerApplicator([(u".*", lambda tensor: constant_(tensor, 0.5))])
        initializer(model)
        value = RegularizerApplicator([(u"", L2Regularizer(1.0))])(model)
        assert value.data.numpy() == 28.75

    def test_regularizer_applicator_respects_regex_matching(self):
        model = torch.nn.Sequential(
                torch.nn.Linear(5, 10),
                torch.nn.Linear(10, 5)
        )
        initializer = InitializerApplicator([(u".*", lambda tensor: constant_(tensor, 1.))])
        initializer(model)
        value = RegularizerApplicator([(u"weight", L2Regularizer(0.5)),
                                       (u"bias", L1Regularizer(1.0))])(model)
        assert value.data.numpy() == 65.0

    def test_from_params(self):
        params = Params({u"regularizers": [(u"conv", u"l1"), (u"linear", {u"type": u"l2", u"alpha": 10})]})
        regularizer_applicator = RegularizerApplicator.from_params(params.pop(u"regularizers"))
        regularizers = regularizer_applicator._regularizers  # pylint: disable=protected-access

        conv = linear = None
        for regex, regularizer in regularizers:
            if regex == u"conv":
                conv = regularizer
            elif regex == u"linear":
                linear = regularizer

        assert isinstance(conv, L1Regularizer)
        assert isinstance(linear, L2Regularizer)
        assert linear.alpha == 10
