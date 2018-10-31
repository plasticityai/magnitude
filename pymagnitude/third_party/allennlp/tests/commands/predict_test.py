# pylint: disable=no-self-use,invalid-name



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import argparse
import csv
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile

import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import JsonDict
from allennlp.common.testing import AllenNlpTestCase
from allennlp.commands import main
from allennlp.commands.predict import Predict
from allennlp.predictors import Predictor, BidafPredictor
from io import open


class TestPredict(AllenNlpTestCase):
    def setUp(self):
        super(TestPredict, self).setUp()
        self.bidaf_model_path = (self.FIXTURES_ROOT / u"bidaf" /
                                 u"serialization" / u"model.tar.gz")
        self.bidaf_data_path = self.FIXTURES_ROOT / u'data' / u'squad.json'
        self.tempdir = pathlib.Path(tempfile.mkdtemp())
        self.infile = self.tempdir / u"inputs.txt"
        self.outfile = self.tempdir / u"outputs.txt"

    def test_add_predict_subparser(self):
        parser = argparse.ArgumentParser(description=u"Testing")
        subparsers = parser.add_subparsers(title=u'Commands', metavar=u'')
        Predict().add_subparser(u'predict', subparsers)

        kebab_args = [u"predict",          # command
                      u"/path/to/archive", # archive
                      u"/dev/null",        # input_file
                      u"--output-file", u"/dev/null",
                      u"--batch-size", u"10",
                      u"--cuda-device", u"0",
                      u"--silent"]

        args = parser.parse_args(kebab_args)

        assert args.func.__name__ == u'_predict'
        assert args.archive_file == u"/path/to/archive"
        assert args.output_file == u"/dev/null"
        assert args.batch_size == 10
        assert args.cuda_device == 0
        assert args.silent

    def test_works_with_known_model(self):
        with open(self.infile, u'w') as f:
            f.write(u"""{"passage": "the seahawks won the super bowl in 2016", """
                    u""" "question": "when did the seahawks win the super bowl?"}\n""")
            f.write(u"""{"passage": "the mariners won the super bowl in 2037", """
                    u""" "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = [u"run.py",      # executable
                    u"predict",     # command
                    unicode(self.bidaf_model_path),
                    unicode(self.infile),     # input_file
                    u"--output-file", unicode(self.outfile),
                    u"--silent"]

        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, u'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == set([u"span_start_logits", u"span_end_logits",
                                          u"passage_question_attention", u"question_tokens",
                                          u"passage_tokens", u"span_start_probs", u"span_end_probs",
                                          u"best_span", u"best_span_str"])

        shutil.rmtree(self.tempdir)

    def test_using_dataset_reader_works_with_known_model(self):

        sys.argv = [u"run.py",      # executable
                    u"predict",     # command
                    unicode(self.bidaf_model_path),
                    unicode(self.bidaf_data_path),     # input_file
                    u"--output-file", unicode(self.outfile),
                    u"--silent",
                    u"--use-dataset-reader"]

        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, u'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 5
        for result in results:
            assert set(result.keys()) == set([u"span_start_logits", u"span_end_logits",
                                          u"passage_question_attention", u"question_tokens",
                                          u"passage_tokens", u"span_start_probs", u"span_end_probs",
                                          u"best_span", u"best_span_str", u"loss"])

        shutil.rmtree(self.tempdir)

    def test_batch_prediction_works_with_known_model(self):
        with open(self.infile, u'w') as f:
            f.write(u"""{"passage": "the seahawks won the super bowl in 2016", """
                    u""" "question": "when did the seahawks win the super bowl?"}\n""")
            f.write(u"""{"passage": "the mariners won the super bowl in 2037", """
                    u""" "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = [u"run.py",  # executable
                    u"predict",  # command
                    unicode(self.bidaf_model_path),
                    unicode(self.infile),  # input_file
                    u"--output-file", unicode(self.outfile),
                    u"--silent",
                    u"--batch-size", u'2']

        main()

        assert os.path.exists(self.outfile)
        with open(self.outfile, u'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        for result in results:
            assert set(result.keys()) == set([u"span_start_logits", u"span_end_logits",
                                          u"passage_question_attention", u"question_tokens",
                                          u"passage_tokens", u"span_start_probs", u"span_end_probs",
                                          u"best_span", u"best_span_str"])

        shutil.rmtree(self.tempdir)

    def test_fails_without_required_args(self):
        sys.argv = [u"run.py",            # executable
                    u"predict",           # command
                    u"/path/to/archive",  # archive, but no input file
                   ]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main()

        assert cm.exception.code == 2  # argparse code for incorrect usage

    def test_can_specify_predictor(self):

        class Bidaf3Predictor(BidafPredictor):
            u"""same as bidaf predictor but with an extra field"""
            def predict_json(self, inputs          )            :
                result = super(Bidaf3Predictor, self).predict_json(inputs)
                result[u"explicit"] = True
                return result

                Bidaf3Predictor = Predictor.register(u'bidaf-explicit')  # pylint: disable=unused-variable(Bidaf3Predictor)

        with open(self.infile, u'w') as f:
            f.write(u"""{"passage": "the seahawks won the super bowl in 2016", """
                    u""" "question": "when did the seahawks win the super bowl?"}\n""")
            f.write(u"""{"passage": "the mariners won the super bowl in 2037", """
                    u""" "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = [u"run.py",      # executable
                    u"predict",     # command
                    unicode(self.bidaf_model_path),
                    unicode(self.infile),     # input_file
                    u"--output-file", unicode(self.outfile),
                    u"--predictor", u"bidaf-explicit",
                    u"--silent"]

        main()
        assert os.path.exists(self.outfile)

        with open(self.outfile, u'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        # Overridden predictor should output extra field
        for result in results:
            assert set(result.keys()) == set([u"span_start_logits", u"span_end_logits",
                                          u"passage_question_attention", u"question_tokens",
                                          u"passage_tokens", u"span_start_probs", u"span_end_probs",
                                          u"best_span", u"best_span_str", u"explicit"])

        shutil.rmtree(self.tempdir)

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / u'testpackage'
        packagedir.mkdir()  # pylint: disable=no-member
        (packagedir / u'__init__.py').touch()  # pylint: disable=no-member

        # And add that directory to the path
        sys.path.insert(0, unicode(self.TEST_DIR))

        # Write out a duplicate predictor there, but registered under a different name.
        from allennlp.predictors import bidaf
        with open(bidaf.__file__) as f:
            code = f.read().replace(u"""@Predictor.register('machine-comprehension')""",
                                    u"""@Predictor.register('duplicate-test-predictor')""")

        with open(os.path.join(packagedir, u'predictor.py'), u'w') as f:
            f.write(code)

        self.infile = os.path.join(self.TEST_DIR, u"inputs.txt")
        self.outfile = os.path.join(self.TEST_DIR, u"outputs.txt")

        with open(self.infile, u'w') as f:
            f.write(u"""{"passage": "the seahawks won the super bowl in 2016", """
                    u""" "question": "when did the seahawks win the super bowl?"}\n""")
            f.write(u"""{"passage": "the mariners won the super bowl in 2037", """
                    u""" "question": "when did the mariners win the super bowl?"}\n""")

        sys.argv = [u"run.py",      # executable
                    u"predict",     # command
                    unicode(self.bidaf_model_path),
                    unicode(self.infile),     # input_file
                    u"--output-file", unicode(self.outfile),
                    u"--predictor", u"duplicate-test-predictor",
                    u"--silent"]

        # Should raise ConfigurationError, because predictor is unknown
        with pytest.raises(ConfigurationError):
            main()

        # But once we include testpackage, it should be known
        sys.argv.extend([u"--include-package", u"testpackage"])
        main()

        assert os.path.exists(self.outfile)

        with open(self.outfile, u'r') as f:
            results = [json.loads(line) for line in f]

        assert len(results) == 2
        # Overridden predictor should output extra field
        for result in results:
            assert set(result.keys()) == set([u"span_start_logits", u"span_end_logits",
                                          u"passage_question_attention", u"question_tokens",
                                          u"passage_tokens", u"span_start_probs", u"span_end_probs",
                                          u"best_span", u"best_span_str"])

        sys.path.remove(unicode(self.TEST_DIR))

    def test_alternative_file_formats(self):
        class BidafCsvPredictor(BidafPredictor):
            u"""same as bidaf predictor but using CSV inputs and outputs"""
            def load_line(self, line     )            :
                reader = csv.reader([line])
                passage, question = next(reader)
                return {u"passage": passage, u"question": question}

            def dump_line(self, outputs          )       :
                output = io.StringIO()
                writer = csv.writer(output)
                row = [outputs[u"span_start_probs"][0],
                       outputs[u"span_end_probs"][0],
                       *outputs[u"best_span"],
                       outputs[u"best_span_str"]]

                writer.writerow(row)
                return output.getvalue()

                BidafCsvPredictor = Predictor.register(u'bidaf-csv')  # pylint: disable=unused-variable(BidafCsvPredictor)

        with open(self.infile, u'w') as f:
            writer = csv.writer(f)
            writer.writerow([u"the seahawks won the super bowl in 2016",
                             u"when did the seahawks win the super bowl?"])
            writer.writerow([u"the mariners won the super bowl in 2037",
                             u"when did the mariners win the super bowl?"])

        sys.argv = [u"run.py",      # executable
                    u"predict",     # command
                    unicode(self.bidaf_model_path),
                    unicode(self.infile),     # input_file
                    u"--output-file", unicode(self.outfile),
                    u"--predictor", u'bidaf-csv',
                    u"--silent"]

        main()
        assert os.path.exists(self.outfile)

        with open(self.outfile, u'r') as f:
            reader = csv.reader(f)
            results = [row for row in reader]

        assert len(results) == 2
        for row in results:
            assert len(row) == 5
            start_prob, end_prob, span_start, span_end, span = row
            for prob in (start_prob, end_prob):
                assert 0 <= float(prob) <= 1
            assert 0 <= int(span_start) <= int(span_end) <= 8
            assert span != u''

        shutil.rmtree(self.tempdir)
