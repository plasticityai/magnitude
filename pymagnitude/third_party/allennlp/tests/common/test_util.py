# pylint: disable=no-self-use,invalid-name



from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import sys
from collections import OrderedDict

import pytest
import torch

from allennlp.common import util
from allennlp.common.testing import AllenNlpTestCase

class Unsanitizable(object):
    pass

class Sanitizable(object):
    def to_json(self):
        return {u"sanitizable": True}

class TestCommonUtils(AllenNlpTestCase):
    def test_group_by_count(self):
        assert util.group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 20) == [[1, 2, 3], [4, 5, 6], [7, 20, 20]]

    def test_lazy_groups_of(self):
        xs = [1, 2, 3, 4, 5, 6, 7]
        groups = util.lazy_groups_of(iter(xs), group_size=3)
        assert next(groups) == [1, 2, 3]
        assert next(groups) == [4, 5, 6]
        assert next(groups) == [7]
        with pytest.raises(StopIteration):
            _ = next(groups)

    def test_pad_sequence_to_length(self):
        assert util.pad_sequence_to_length([1, 2, 3], 5) == [1, 2, 3, 0, 0]
        assert util.pad_sequence_to_length([1, 2, 3], 5, default_value=lambda: 2) == [1, 2, 3, 2, 2]
        assert util.pad_sequence_to_length([1, 2, 3], 5, padding_on_right=False) == [0, 0, 1, 2, 3]

    def test_namespace_match(self):
        assert util.namespace_match(u"*tags", u"tags")
        assert util.namespace_match(u"*tags", u"passage_tags")
        assert util.namespace_match(u"*tags", u"question_tags")
        assert util.namespace_match(u"tokens", u"tokens")
        assert not util.namespace_match(u"tokens", u"stemmed_tokens")

    def test_sanitize(self):
        assert util.sanitize(torch.Tensor([1, 2])) == [1, 2]
        assert util.sanitize(torch.LongTensor([1, 2])) == [1, 2]

        with pytest.raises(ValueError):
            util.sanitize(Unsanitizable())

        assert util.sanitize(Sanitizable()) == {u"sanitizable": True}

    def test_import_submodules(self):
        # pylint: disable=no-member
        (self.TEST_DIR / u'mymodule').mkdir()
        (self.TEST_DIR / u'mymodule' / u'__init__.py').touch()
        (self.TEST_DIR / u'mymodule' / u'submodule').mkdir()
        (self.TEST_DIR / u'mymodule' / u'submodule' / u'__init__.py').touch()
        (self.TEST_DIR / u'mymodule' / u'submodule' / u'subsubmodule.py').touch()

        sys.path.insert(0, unicode(self.TEST_DIR))
        assert u'mymodule' not in sys.modules
        assert u'mymodule.submodule' not in sys.modules

        util.import_submodules(u'mymodule')

        assert u'mymodule' in sys.modules
        assert u'mymodule.submodule' in sys.modules
        assert u'mymodule.submodule.subsubmodule' in sys.modules

        sys.path.remove(unicode(self.TEST_DIR))


    def test_get_frozen_and_tunable_parameter_names(self):
        model = torch.nn.Sequential(OrderedDict([
                (u'conv', torch.nn.Conv1d(5, 5, 5)),
                (u'linear', torch.nn.Linear(5, 10)),
                ]))
        named_parameters = dict(model.named_parameters())
        named_parameters[u'linear.weight'].requires_grad_(False)
        named_parameters[u'linear.bias'].requires_grad_(False)
        frozen_parameter_names, tunable_parameter_names =\
                       util.get_frozen_and_tunable_parameter_names(model)
        assert set(frozen_parameter_names) == set([u'linear.weight', u'linear.bias'])
        assert set(tunable_parameter_names) == set([u'conv.weight', u'conv.bias'])
