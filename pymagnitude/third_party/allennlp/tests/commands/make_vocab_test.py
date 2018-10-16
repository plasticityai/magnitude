# pylint: disable=invalid-name,no-self-use



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import argparse
import os

import pytest

from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.make_vocab import MakeVocab, make_vocab_from_args, make_vocab_from_params
from allennlp.data import Vocabulary
from allennlp.common.checks import ConfigurationError
from io import open

class TestMakeVocab(AllenNlpTestCase):
    def setUp(self):
        super(TestMakeVocab, self).setUp()

        self.params = Params({
                u"model": {
                        u"type": u"simple_tagger",
                        u"text_field_embedder": {
                                u"tokens": {
                                        u"type": u"embedding",
                                        u"embedding_dim": 5
                                }
                        },
                        u"encoder": {
                                u"type": u"lstm",
                                u"input_size": 5,
                                u"hidden_size": 7,
                                u"num_layers": 2
                        }
                },
                u"dataset_reader": {u"type": u"sequence_tagging"},
                u"train_data_path": self.FIXTURES_ROOT / u'data' / u'sequence_tagging.tsv',
                u"validation_data_path": self.FIXTURES_ROOT / u'data' / u'sequence_tagging.tsv',
                u"iterator": {u"type": u"basic", u"batch_size": 2},
                u"trainer": {
                        u"num_epochs": 2,
                        u"optimizer": u"adam"
                }
        })

    def test_make_vocab_doesnt_overwrite_vocab(self):
        vocab_path = self.TEST_DIR / u'vocabulary'
        os.mkdir(vocab_path)
        # Put something in the vocab directory
        with open(vocab_path / u"test.txt", u"a+") as open_file:
            open_file.write(u"test")
        # It should raise error if vocab dir is non-empty
        with pytest.raises(ConfigurationError):
            make_vocab_from_params(self.params, self.TEST_DIR)

    def test_make_vocab_succeeds_without_vocabulary_key(self):
        make_vocab_from_params(self.params, self.TEST_DIR)

    def test_make_vocab_makes_vocab(self):
        vocab_path = self.TEST_DIR / u'vocabulary'

        make_vocab_from_params(self.params, self.TEST_DIR)

        vocab_files = os.listdir(vocab_path)
        assert set(vocab_files) == set([u'labels.txt', u'non_padded_namespaces.txt', u'tokens.txt'])

        with open(vocab_path / u'tokens.txt') as f:
            tokens = [line.strip() for line in f]

        tokens.sort()
        assert tokens == [u'.', u'@@UNKNOWN@@', u'animals', u'are', u'birds', u'cats', u'dogs', u'snakes']

        with open(vocab_path / u'labels.txt') as f:
            labels = [line.strip() for line in f]

        labels.sort()
        assert labels == [u'N', u'V']

    def test_make_vocab_makes_vocab_with_config(self):
        vocab_path = self.TEST_DIR / u'vocabulary'

        self.params[u'vocabulary'] = {}
        self.params[u'vocabulary'][u'min_count'] = {u"tokens" : 3}

        make_vocab_from_params(self.params, self.TEST_DIR)

        vocab_files = os.listdir(vocab_path)
        assert set(vocab_files) == set([u'labels.txt', u'non_padded_namespaces.txt', u'tokens.txt'])

        with open(vocab_path / u'tokens.txt') as f:
            tokens = [line.strip() for line in f]

        tokens.sort()
        assert tokens == [u'.', u'@@UNKNOWN@@', u'animals', u'are']

        with open(vocab_path / u'labels.txt') as f:
            labels = [line.strip() for line in f]

        labels.sort()
        assert labels == [u'N', u'V']

    def test_make_vocab_with_extension(self):
        existing_serialization_dir = self.TEST_DIR / u'existing'
        extended_serialization_dir = self.TEST_DIR / u'extended'
        existing_vocab_path = existing_serialization_dir / u'vocabulary'
        extended_vocab_path = extended_serialization_dir / u'vocabulary'

        vocab = Vocabulary()
        vocab.add_token_to_namespace(u'some_weird_token_1', namespace=u'tokens')
        vocab.add_token_to_namespace(u'some_weird_token_2', namespace=u'tokens')
        os.makedirs(existing_serialization_dir, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)

        self.params[u'vocabulary'] = {}
        self.params[u'vocabulary'][u'directory_path'] = existing_vocab_path
        self.params[u'vocabulary'][u'extend'] = True
        self.params[u'vocabulary'][u'min_count'] = {u"tokens" : 3}
        make_vocab_from_params(self.params, extended_serialization_dir)

        vocab_files = os.listdir(extended_vocab_path)
        assert set(vocab_files) == set([u'labels.txt', u'non_padded_namespaces.txt', u'tokens.txt'])

        with open(extended_vocab_path / u'tokens.txt') as f:
            tokens = [line.strip() for line in f]

        assert tokens[0] == u'@@UNKNOWN@@'
        assert tokens[1] == u'some_weird_token_1'
        assert tokens[2] == u'some_weird_token_2'

        tokens.sort()
        assert tokens == [u'.', u'@@UNKNOWN@@', u'animals', u'are',
                          u'some_weird_token_1', u'some_weird_token_2']

        with open(extended_vocab_path / u'labels.txt') as f:
            labels = [line.strip() for line in f]

        labels.sort()
        assert labels == [u'N', u'V']

    def test_make_vocab_without_extension(self):
        existing_serialization_dir = self.TEST_DIR / u'existing'
        extended_serialization_dir = self.TEST_DIR / u'extended'
        existing_vocab_path = existing_serialization_dir / u'vocabulary'
        extended_vocab_path = extended_serialization_dir / u'vocabulary'

        vocab = Vocabulary()
        vocab.add_token_to_namespace(u'some_weird_token_1', namespace=u'tokens')
        vocab.add_token_to_namespace(u'some_weird_token_2', namespace=u'tokens')
        # if extend is False, its users responsibility to make sure that dataset instances
        # will be indexible by provided vocabulary. At least @@UNKNOWN@@ should be present in
        # namespace for which there could be OOV entries seen in dataset during indexing.
        # For `tokens` ns, new words will be seen but `tokens` has @@UNKNOWN@@ token.
        # but for 'labels' ns, there is no @@UNKNOWN@@ so required to add 'N', 'V' upfront.
        vocab.add_token_to_namespace(u'N', namespace=u'labels')
        vocab.add_token_to_namespace(u'V', namespace=u'labels')
        os.makedirs(existing_serialization_dir, exist_ok=True)
        vocab.save_to_files(existing_vocab_path)

        self.params[u'vocabulary'] = {}
        self.params[u'vocabulary'][u'directory_path'] = existing_vocab_path
        self.params[u'vocabulary'][u'extend'] = False
        make_vocab_from_params(self.params, extended_serialization_dir)

        with open(extended_vocab_path / u'tokens.txt') as f:
            tokens = [line.strip() for line in f]

        assert tokens[0] == u'@@UNKNOWN@@'
        assert tokens[1] == u'some_weird_token_1'
        assert tokens[2] == u'some_weird_token_2'
        assert len(tokens) == 3

    def test_make_vocab_args(self):
        parser = argparse.ArgumentParser(description=u"Testing")
        subparsers = parser.add_subparsers(title=u'Commands', metavar=u'')
        MakeVocab().add_subparser(u'make-vocab', subparsers)
        for serialization_arg in [u"-s", u"--serialization-dir"]:
            raw_args = [u"make-vocab", u"path/to/params", serialization_arg, u"serialization_dir"]
            args = parser.parse_args(raw_args)
            assert args.func == make_vocab_from_args
            assert args.param_path == u"path/to/params"
            assert args.serialization_dir == u"serialization_dir"
