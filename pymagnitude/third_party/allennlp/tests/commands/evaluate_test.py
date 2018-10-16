# pylint: disable=invalid-name,no-self-use



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import argparse
import json

from flaky import flaky

from allennlp.commands.evaluate import evaluate_from_args, Evaluate
from allennlp.common.testing import AllenNlpTestCase
from io import open


class TestEvaluate(AllenNlpTestCase):
    def setUp(self):
        super(TestEvaluate, self).setUp()

        self.parser = argparse.ArgumentParser(description=u"Testing")
        subparsers = self.parser.add_subparsers(title=u'Commands', metavar=u'')
        Evaluate().add_subparser(u'evaluate', subparsers)

    @flaky
    def test_evaluate_from_args(self):
        kebab_args = [u"evaluate", unicode(self.FIXTURES_ROOT / u"bidaf" / u"serialization" / u"model.tar.gz"),
                      unicode(self.FIXTURES_ROOT / u"data" / u"squad.json"),
                      u"--cuda-device", u"-1"]

        args = self.parser.parse_args(kebab_args)
        metrics = evaluate_from_args(args)
        assert list(metrics.keys()) == set([u'span_acc', u'end_acc', u'start_acc', u'em', u'f1'])

    def test_output_file_evaluate_from_args(self):
        output_file = unicode(self.TEST_DIR / u"metrics.json")
        kebab_args = [u"evaluate", unicode(self.FIXTURES_ROOT / u"bidaf" / u"serialization" / u"model.tar.gz"),
                      unicode(self.FIXTURES_ROOT / u"data" / u"squad.json"),
                      u"--cuda-device", u"-1",
                      u"--output-file", output_file]
        args = self.parser.parse_args(kebab_args)
        computed_metrics = evaluate_from_args(args)
        with open(output_file, u'r') as file:
            saved_metrics = json.load(file)
        assert computed_metrics == saved_metrics
