# pylint: disable=no-self-use,invalid-name


from __future__ import with_statement
from __future__ import absolute_import
import pytest

from allennlp.common.configuration import configure, Config, BASE_CONFIG
from allennlp.common.testing import AllenNlpTestCase


class TestConfiguration(AllenNlpTestCase):
    def test_configure_top_level(self):
        config = configure()

        assert config == BASE_CONFIG

    def test_abstract_base_class(self):
        config = configure(u'allennlp.data.dataset_readers.dataset_reader.DatasetReader')

        assert isinstance(config, list)
        assert u'allennlp.data.dataset_readers.snli.SnliReader' in config

    def test_specific_subclass(self):
        config = configure(u'allennlp.data.dataset_readers.semantic_role_labeling.SrlReader')
        assert isinstance(config, Config)

        items = dict((item.name, item) for item in config.items)

        assert len(items) == 3

        assert u'token_indexers' in items
        token_indexers = items[u'token_indexers']
        assert token_indexers.default_value is None

        assert u'domain_identifier' in items
        domain_identifier = items[u'domain_identifier']
        assert domain_identifier.annotation == unicode
        assert domain_identifier.default_value is None

        assert u'lazy' in items
        lazy = items[u'lazy']
        assert lazy.annotation == bool
        assert not lazy.default_value

    def test_errors(self):
        with pytest.raises(ModuleNotFoundError):
            configure(u'allennlp.non_existent_module.SomeClass')

        with pytest.raises(AttributeError):
            configure(u'allennlp.data.dataset_readers.NonExistentDatasetReader')
