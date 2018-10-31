# pylint: disable=no-self-use,invalid-name
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import pytest

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from io import open

# Skip this one, it's an expensive test.
class TestOpenaiTransformerEmbedderLarge(ModelTestCase):
    def setUp(self):
        super(TestOpenaiTransformerEmbedderLarge, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / u'openai_transformer' / u'config_large.jsonnet',
                          self.FIXTURES_ROOT / u'data' / u'conll2003.txt')

    def test_tagger_with_openai_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_openai_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict[u'tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace=u"labels")
                assert tag in set([u'O', u'I-ORG', u'I-PER', u'I-LOC'])


TestOpenaiTransformerEmbedderLarge =  Skip this one, it's an expensive test.
@pytest.mark.skip()(TestOpenaiTransformerEmbedderLarge)

class TestOpenaiTransformerEmbedderSmall(ModelTestCase):
    def setUp(self):
        super(TestOpenaiTransformerEmbedderSmall, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / u'openai_transformer' / u'config_small.jsonnet',
                          self.FIXTURES_ROOT / u'data' / u'conll2003.txt')

    def test_tagger_with_openai_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_openai_token_embedder_forward_pass_runs_correctly(self):
        dataset = Batch(self.instances)
        dataset.index_instances(self.vocab)
        training_tensors = dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        tags = output_dict[u'tags']
        assert len(tags) == 2
        assert len(tags[0]) == 7
        assert len(tags[1]) == 7
        for example_tags in tags:
            for tag_id in example_tags:
                tag = self.model.vocab.get_token_from_index(tag_id, namespace=u"labels")
                assert tag in set([u'O', u'I-ORG', u'I-PER', u'I-LOC'])


def create_small_test_fixture(output_dir      = u'/tmp')        :
    u"""
    This is how I created the transformer_model.tar.gz.
    After running this, go to the specified output dir and run

        tar -czvf transformer_model.tar.gz model/

    In case you need to regenerate the fixture for some reason.
    """
    import json
    import pathlib
    from allennlp.modules.openai_transformer import OpenaiTransformer

    model_dir = pathlib.Path(output_dir) / u'model'
    model_dir.mkdir(exist_ok=True)  # pylint: disable=no-member

    symbols = [u"e", u"w", u"o", u"wo", u"."]
    byte_pairs = [(sym1, sym2 + end)
                  for sym1 in symbols        # prefer earlier first symbol
                  for sym2 in symbols        # if tie, prefer earlier second symbol
                  for end in (u'</w>', u'')]   # if tie, prefer ending a word
    encoding = dict(("{sym1}{sym2}", idx) for idx, (sym1, sym2) in enumerate(byte_pairs))
    encoding[u"<unk>"] = 0

    with open(model_dir / u'encoder_bpe.json', u'w') as encoder_file:
        json.dump(encoding, encoder_file)

    with open(model_dir / u'vocab.bpe', u'w') as bpe_file:
        bpe_file.write(u"#version 0.0\n")
        for sym1, sym2 in byte_pairs:
            bpe_file.write("{sym1} {sym2}\n")
        bpe_file.write(u"\n")

    transformer = OpenaiTransformer(embedding_dim=10, num_heads=2, num_layers=2, vocab_size=(50 + 50), n_ctx=50)
    transformer.dump_weights(output_dir, num_pieces=2)
