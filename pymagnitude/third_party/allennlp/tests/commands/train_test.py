# pylint: disable=invalid-name,no-self-use



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import argparse
#typing
import os
import shutil
import re

import pytest
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.train import Train, train_model, train_model_from_args
from allennlp.data import DatasetReader, Instance
from io import open

SEQUENCE_TAGGING_DATA_PATH = unicode(AllenNlpTestCase.FIXTURES_ROOT / u'data' / u'sequence_tagging.tsv')

class TestTrain(AllenNlpTestCase):

    def test_train_model(self):
        params = lambda: Params({
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
                u"train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"iterator": {u"type": u"basic", u"batch_size": 2},
                u"trainer": {
                        u"num_epochs": 2,
                        u"optimizer": u"adam"
                }
        })

        train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, u'test_train_model'))

        # It's OK if serialization dir exists but is empty:
        serialization_dir2 = os.path.join(self.TEST_DIR, u'empty_directory')
        assert not os.path.exists(serialization_dir2)
        os.makedirs(serialization_dir2)
        train_model(params(), serialization_dir=serialization_dir2)

        # It's not OK if serialization dir exists and has junk in it non-empty:
        serialization_dir3 = os.path.join(self.TEST_DIR, u'non_empty_directory')
        assert not os.path.exists(serialization_dir3)
        os.makedirs(serialization_dir3)
        with open(os.path.join(serialization_dir3, u'README.md'), u'w') as f:
            f.write(u"TEST")

        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=serialization_dir3)

        # It's also not OK if serialization dir is a real serialization dir:
        with pytest.raises(ConfigurationError):
            train_model(params(), serialization_dir=os.path.join(self.TEST_DIR, u'test_train_model'))

    def test_error_is_throw_when_cuda_device_is_not_available(self):
        params = Params({
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
                u"train_data_path": u'tests/fixtures/data/sequence_tagging.tsv',
                u"validation_data_path": u'tests/fixtures/data/sequence_tagging.tsv',
                u"iterator": {u"type": u"basic", u"batch_size": 2},
                u"trainer": {
                        u"num_epochs": 2,
                        u"cuda_device": torch.cuda.device_count(),
                        u"optimizer": u"adam"
                }
        })

        with pytest.raises(ConfigurationError,
                           message=u"Experiment specified a GPU but none is available;"
                                   u" if you want to run on CPU use the override"
                                   u" 'trainer.cuda_device=-1' in the json config file."):
            train_model(params, serialization_dir=os.path.join(self.TEST_DIR, u'test_train_model'))

    def test_train_with_test_set(self):
        params = Params({
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
                u"train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"test_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"evaluate_on_test": True,
                u"iterator": {u"type": u"basic", u"batch_size": 2},
                u"trainer": {
                        u"num_epochs": 2,
                        u"optimizer": u"adam"
                }
        })

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, u'train_with_test_set'))

    def test_train_args(self):
        parser = argparse.ArgumentParser(description=u"Testing")
        subparsers = parser.add_subparsers(title=u'Commands', metavar=u'')
        Train().add_subparser(u'train', subparsers)

        for serialization_arg in [u"-s", u"--serialization-dir"]:
            raw_args = [u"train", u"path/to/params", serialization_arg, u"serialization_dir"]

            args = parser.parse_args(raw_args)

            assert args.func == train_model_from_args
            assert args.param_path == u"path/to/params"
            assert args.serialization_dir == u"serialization_dir"

        # config is required
        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            args = parser.parse_args([u"train", u"-s", u"serialization_dir"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

        # serialization dir is required
        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            args = parser.parse_args([u"train", u"path/to/params"])
            assert cm.exception.code == 2  # argparse code for incorrect usage

class LazyFakeReader(DatasetReader):
    # pylint: disable=abstract-method
    def __init__(self)        :
        super(LazyFakeReader, self).__init__(lazy=True)
        self.reader = DatasetReader.from_params(Params({u'type': u'sequence_tagging'}))

    def _read(self, file_path     )                      :
        u"""
        Reads some data from the `file_path` and returns the instances.
        """
        return self.reader.read(file_path)


LazyFakeReader = DatasetReader.register(u'lazy-test')(LazyFakeReader)

class TestTrainOnLazyDataset(AllenNlpTestCase):
    def test_train_model(self):
        params = Params({
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
                u"dataset_reader": {u"type": u"lazy-test"},
                u"train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"iterator": {u"type": u"basic", u"batch_size": 2},
                u"trainer": {
                        u"num_epochs": 2,
                        u"optimizer": u"adam"
                }
        })

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, u'train_lazy_model'))

    def test_train_with_test_set(self):
        params = Params({
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
                u"dataset_reader": {u"type": u"lazy-test"},
                u"train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"test_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"evaluate_on_test": True,
                u"iterator": {u"type": u"basic", u"batch_size": 2},
                u"trainer": {
                        u"num_epochs": 2,
                        u"optimizer": u"adam"
                }
        })

        train_model(params, serialization_dir=os.path.join(self.TEST_DIR, u'lazy_test_set'))

    def test_train_nograd_regex(self):
        params_get = lambda: Params({
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
                u"train_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"validation_data_path": SEQUENCE_TAGGING_DATA_PATH,
                u"iterator": {u"type": u"basic", u"batch_size": 2},
                u"trainer": {
                        u"num_epochs": 2,
                        u"optimizer": u"adam"
                }
        })
        serialization_dir = os.path.join(self.TEST_DIR, u'test_train_nograd')
        regex_lists = [[],
                       [u".*text_field_embedder.*"],
                       [u".*text_field_embedder.*", u".*encoder.*"]]
        for regex_list in regex_lists:
            params = params_get()
            params[u"trainer"][u"no_grad"] = regex_list
            shutil.rmtree(serialization_dir, ignore_errors=True)
            model = train_model(params, serialization_dir=serialization_dir)
            # If regex is matched, parameter name should have requires_grad False
            # Or else True
            for name, parameter in model.named_parameters():
                if any(re.search(regex, name) for regex in regex_list):
                    assert not parameter.requires_grad
                else:
                    assert parameter.requires_grad
        # If all parameters have requires_grad=False, then error.
        params = params_get()
        params[u"trainer"][u"no_grad"] = [u"*"]
        shutil.rmtree(serialization_dir, ignore_errors=True)
        with pytest.raises(Exception) as _:
            model = train_model(params, serialization_dir=serialization_dir)
