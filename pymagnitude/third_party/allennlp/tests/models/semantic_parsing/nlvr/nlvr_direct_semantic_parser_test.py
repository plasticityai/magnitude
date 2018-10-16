

from __future__ import division
from __future__ import absolute_import
from allennlp.common.testing import ModelTestCase


class NlvrDirectSemanticParserTest(ModelTestCase):
    def setUp(self):
        super(NlvrDirectSemanticParserTest, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / u"semantic_parsing" /
                          u"nlvr_direct_semantic_parser" / u"experiment.json",
                          self.FIXTURES_ROOT / u"data" / u"nlvr" / u"sample_processed_data.jsonl")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
