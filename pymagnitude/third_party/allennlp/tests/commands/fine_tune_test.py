# pylint: disable=invalid-name,no-self-use



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import argparse
import re
import shutil

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands.fine_tune import FineTune, fine_tune_model_from_file_paths,\
                               fine_tune_model_from_args, fine_tune_model
from allennlp.common.params import Params
from allennlp.models import load_archive

class TestFineTune(AllenNlpTestCase):
    def setUp(self):
        super(TestFineTune, self).setUp()
        self.model_archive = unicode(self.FIXTURES_ROOT / u'decomposable_attention' / u'serialization' / u'model.tar.gz')
        self.config_file = unicode(self.FIXTURES_ROOT / u'decomposable_attention' / u'experiment.json')
        self.serialization_dir = unicode(self.TEST_DIR / u'fine_tune')

        self.parser = argparse.ArgumentParser(description=u"Testing")
        subparsers = self.parser.add_subparsers(title=u'Commands', metavar=u'')
        FineTune().add_subparser(u'fine-tune', subparsers)


    def test_fine_tune_model_runs_from_file_paths(self):
        fine_tune_model_from_file_paths(model_archive_path=self.model_archive,
                                        config_file=self.config_file,
                                        serialization_dir=self.serialization_dir)

    def test_fine_tune_does_not_expand_vocab_by_default(self):
        params = Params.from_file(self.config_file)
        # snli2 has a new token in it
        params[u"train_data_path"] = unicode(self.FIXTURES_ROOT / u'data' / u'snli2.jsonl')

        model = load_archive(self.model_archive).model

        # By default, no vocab expansion.
        fine_tune_model(model, params, self.serialization_dir)

    def test_fine_tune_runtime_errors_with_vocab_expansion(self):
        params = Params.from_file(self.config_file)
        params[u"train_data_path"] = unicode(self.FIXTURES_ROOT / u'data' / u'snli2.jsonl')

        model = load_archive(self.model_archive).model

        # If we do vocab expansion, we get a runtime error because of the embedding.
        with pytest.raises(RuntimeError):
            fine_tune_model(model, params, self.serialization_dir, extend_vocab=True)

    def test_fine_tune_runs_from_parser_arguments(self):
        raw_args = [u"fine-tune",
                    u"-m", self.model_archive,
                    u"-c", self.config_file,
                    u"-s", self.serialization_dir]

        args = self.parser.parse_args(raw_args)

        assert args.func == fine_tune_model_from_args
        assert args.model_archive == self.model_archive
        assert args.config_file == self.config_file
        assert args.serialization_dir == self.serialization_dir
        fine_tune_model_from_args(args)

    def test_fine_tune_fails_without_required_args(self):
        # Configuration file is required.
        with self.assertRaises(SystemExit) as context:
            self.parser.parse_args([u"fine-tune", u"-m", u"path/to/archive", u"-s", u"serialization_dir"])
            assert context.exception.code == 2  # argparse code for incorrect usage

        # Serialization dir is required.
        with self.assertRaises(SystemExit) as context:
            self.parser.parse_args([u"fine-tune", u"-m", u"path/to/archive", u"-c", u"path/to/config"])
            assert context.exception.code == 2  # argparse code for incorrect usage

        # Model archive is required.
        with self.assertRaises(SystemExit) as context:
            self.parser.parse_args([u"fine-tune", u"-s", u"serialization_dir", u"-c", u"path/to/config"])
            assert context.exception.code == 2  # argparse code for incorrect usage

    def test_fine_tune_nograd_regex(self):
        original_model = load_archive(self.model_archive).model
        name_parameters_original = dict(original_model.named_parameters())
        regex_lists = [[],
                       [u".*attend_feedforward.*", u".*token_embedder.*"],
                       [u".*compare_feedforward.*"]]
        for regex_list in regex_lists:
            params = Params.from_file(self.config_file)
            params[u"trainer"][u"no_grad"] = regex_list
            shutil.rmtree(self.serialization_dir, ignore_errors=True)
            tuned_model = fine_tune_model(model=original_model,
                                          params=params,
                                          serialization_dir=self.serialization_dir)
            # If regex is matched, parameter name should have requires_grad False
            # If regex is matched, parameter name should have same requires_grad
            # as the originally loaded model
            for name, parameter in tuned_model.named_parameters():
                if any(re.search(regex, name) for regex in regex_list):
                    assert not parameter.requires_grad
                else:
                    assert parameter.requires_grad\
                    == name_parameters_original[name].requires_grad
        # If all parameters have requires_grad=False, then error.
        with pytest.raises(Exception) as _:
            params = Params.from_file(self.config_file)
            params[u"trainer"][u"no_grad"] = [u"*"]
            shutil.rmtree(self.serialization_dir, ignore_errors=True)
            tuned_model = fine_tune_model(model=original_model,
                                          params=params,
                                          serialization_dir=self.serialization_dir)
