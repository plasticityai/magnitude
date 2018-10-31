# pylint: disable=no-self-use


from __future__ import division
from __future__ import absolute_import
import os
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.training.metrics.wikitables_accuracy import SEMPRE_ABBREVIATIONS_PATH, SEMPRE_GRAMMAR_PATH

class WikiTablesErmSemanticParserTest(ModelTestCase):
    def setUp(self):
        self.should_remove_sempre_abbreviations = not os.path.exists(SEMPRE_ABBREVIATIONS_PATH)
        self.should_remove_sempre_grammar = not os.path.exists(SEMPRE_GRAMMAR_PATH)

        # The model tests are run with respect to the module root, so check if abbreviations
        # and grammar already exist there (since we want to clean up module root after test)
        self.module_root_abbreviations_path = self.MODULE_ROOT / u"data" / u"abbreviations.tsv"
        self.module_root_grammar_path = self.MODULE_ROOT / u"data" / u"grow.grammar"
        self.should_remove_root_sempre_abbreviations = not os.path.exists(self.module_root_abbreviations_path)
        self.should_remove_root_sempre_grammar = not os.path.exists(self.module_root_grammar_path)

        super(WikiTablesErmSemanticParserTest, self).setUp()
        self.set_up_model(unicode(self.FIXTURES_ROOT / u"semantic_parsing" / u"wikitables" / u"experiment-erm.json"),
                          unicode(self.FIXTURES_ROOT / u"data" / u"wikitables" / u"sample_data.examples"))

    def tearDown(self):
        super(WikiTablesErmSemanticParserTest, self).tearDown()
        # We don't want to leave generated files around just from running tests...
        if self.should_remove_sempre_abbreviations and os.path.exists(SEMPRE_ABBREVIATIONS_PATH):
            os.remove(SEMPRE_ABBREVIATIONS_PATH)
        if self.should_remove_sempre_grammar and os.path.exists(SEMPRE_GRAMMAR_PATH):
            os.remove(SEMPRE_GRAMMAR_PATH)
        if self.should_remove_root_sempre_abbreviations and os.path.exists(self.module_root_abbreviations_path):
            os.remove(self.module_root_abbreviations_path)
        if self.should_remove_root_sempre_grammar and os.path.exists(self.module_root_grammar_path):
            os.remove(self.module_root_grammar_path)

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

WikiTablesErmSemanticParserTest = pytest.mark.java(WikiTablesErmSemanticParserTest)
