# pylint: disable=no-self-use,invalid-name



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import filecmp
import json
import os
import pathlib
import tarfile

import torch

from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from io import open


class TestElmoTokenEmbedder(ModelTestCase):
    def setUp(self):
        super(TestElmoTokenEmbedder, self).setUp()
        self.set_up_model(self.FIXTURES_ROOT / u'elmo' / u'config' / u'characters_token_embedder.json',
                          self.FIXTURES_ROOT / u'data' / u'conll2003.txt')

    def test_tagger_with_elmo_token_embedder_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    def test_tagger_with_elmo_token_embedder_forward_pass_runs_correctly(self):
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

    def test_file_archiving(self):
        # This happens to be a good place to test auxiliary file archiving.
        # Train the model
        params = Params.from_file(self.FIXTURES_ROOT / u'elmo' / u'config' / u'characters_token_embedder.json')
        serialization_dir = os.path.join(self.TEST_DIR, u'serialization')
        train_model(params, serialization_dir)

        # Inspect the archive
        archive_file = os.path.join(serialization_dir, u'model.tar.gz')
        unarchive_dir = os.path.join(self.TEST_DIR, u'unarchive')
        with tarfile.open(archive_file, u'r:gz') as archive:
            archive.extractall(unarchive_dir)

        # It should contain `files_to_archive.json`
        fta_file = os.path.join(unarchive_dir, u'files_to_archive.json')
        assert os.path.exists(fta_file)

        # Which should properly contain { flattened_key -> original_filename }
        with open(fta_file) as fta:
            files_to_archive = json.loads(fta.read())

        assert files_to_archive == {
                u'model.text_field_embedder.token_embedders.elmo.options_file':
                        unicode(pathlib.Path(u'allennlp') / u'tests' / u'fixtures' / u'elmo' / u'options.json'),
                u'model.text_field_embedder.token_embedders.elmo.weight_file':
                        unicode(pathlib.Path(u'allennlp') / u'tests' / u'fixtures' / u'elmo' / u'lm_weights.hdf5'),
        }

        # Check that the unarchived contents of those files match the original contents.
        for key, original_filename in list(files_to_archive.items()):
            new_filename = os.path.join(unarchive_dir, u"fta", key)
            assert filecmp.cmp(original_filename, new_filename)

    def test_forward_works_with_projection_layer(self):
        params = Params({
                u'options_file': self.FIXTURES_ROOT / u'elmo' / u'options.json',
                u'weight_file': self.FIXTURES_ROOT / u'elmo' / u'lm_weights.hdf5',
                u'projection_dim': 20
                })
        word1 = [0] * 50
        word2 = [0] * 50
        word1[0] = 6
        word1[1] = 5
        word1[2] = 4
        word1[3] = 3
        word2[0] = 3
        word2[1] = 2
        word2[2] = 1
        word2[3] = 0
        embedding_layer = ElmoTokenEmbedder.from_params(vocab=None, params=params)
        input_tensor = torch.LongTensor([[word1, word2]])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 2, 20)

        input_tensor = torch.LongTensor([[[word1]]])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 1, 1, 20)
