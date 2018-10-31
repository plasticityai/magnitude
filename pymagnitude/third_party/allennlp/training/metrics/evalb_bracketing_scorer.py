



from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
#typing
import os
import tempfile
import subprocess
import shutil

#overrides
from nltk import Tree

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric
from io import open

DEFAULT_EVALB_DIR = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, u"tools", u"EVALB"))

class EvalbBracketingScorer(Metric):
    u"""
    This class uses the external EVALB software for computing a broad range of metrics
    on parse trees. Here, we use it to compute the Precision, Recall and F1 metrics.
    You can download the source for EVALB from here: <http://nlp.cs.nyu.edu/evalb/>.

    Note that this software is 20 years old. In order to compile it on modern hardware,
    you may need to remove an ``include <malloc.h>`` statement in ``evalb.c`` before it
    will compile.

    AllenNLP contains the EVALB software, but you will need to compile it yourself
    before using it because the binary it generates is system depenedent. To build it,
    run ``make`` inside the ``allennlp/tools/EVALB`` directory.

    Note that this metric reads and writes from disk quite a bit. You probably don't
    want to include it in your training loop; instead, you should calculate this on
    a validation set only.

    Parameters
    ----------
    evalb_directory_path : ``str``, required.
        The directory containing the EVALB executable.
    evalb_param_filename: ``str``, optional (default = "COLLINS.prm")
        The relative name of the EVALB configuration file used when scoring the trees.
        By default, this uses the COLLINS.prm configuration file which comes with EVALB.
        This configuration ignores POS tags and some punctuation labels.
    """
    def __init__(self,
                 evalb_directory_path      = DEFAULT_EVALB_DIR,
                 evalb_param_filename      = u"COLLINS.prm")        :
        self._evalb_directory_path = evalb_directory_path
        self._evalb_program_path = os.path.join(evalb_directory_path, u"evalb")
        self._evalb_param_path = os.path.join(evalb_directory_path, evalb_param_filename)


        self._header_line = [u'ID', u'Len.', u'Stat.', u'Recal', u'Prec.', u'Bracket',
                             u'gold', u'test', u'Bracket', u'Words', u'Tags', u'Accracy']

        self._correct_predicted_brackets = 0.0
        self._gold_brackets = 0.0
        self._predicted_brackets = 0.0

    #overrides
    def __call__(self, predicted_trees            , gold_trees            )        : # type: ignore
        u"""
        Parameters
        ----------
        predicted_trees : ``List[Tree]``
            A list of predicted NLTK Trees to compute score for.
        gold_trees : ``List[Tree]``
            A list of gold NLTK Trees to use as a reference.
        """
        if not os.path.exists(self._evalb_program_path):
            compile_command = (u"python -c 'from allennlp.training.metrics import EvalbBracketingScorer; "
                               u"EvalbBracketingScorer.compile_evalb()'")
            raise ConfigurationError(u"You must compile the EVALB scorer before using it."
                                     u" Run 'make' in the '{}' directory or run: {}".format(
                                             self._evalb_program_path, compile_command))
        tempdir = tempfile.mkdtemp()
        gold_path = os.path.join(tempdir, u"gold.txt")
        predicted_path = os.path.join(tempdir, u"predicted.txt")
        output_path = os.path.join(tempdir, u"output.txt")
        with open(gold_path, u"w") as gold_file:
            for tree in gold_trees:
                gold_file.write("{tree.pformat(margin=1000000)}\n")

        with open(predicted_path, u"w") as predicted_file:
            for tree in predicted_trees:
                predicted_file.write("{tree.pformat(margin=1000000)}\n")

        command = "{self._evalb_program_path} -p {self._evalb_param_path} "\
                  "{gold_path} {predicted_path} > {output_path}"
        subprocess.run(command, shell=True, check=True)

        with open(output_path) as infile:
            for line in infile:
                stripped = line.strip().split()
                if len(stripped) == 12 and stripped != self._header_line:
                    # This line contains results for a single tree.
                    numeric_line = [float(x) for x in stripped]
                    self._correct_predicted_brackets += numeric_line[5]
                    self._gold_brackets += numeric_line[6]
                    self._predicted_brackets += numeric_line[7]

        shutil.rmtree(tempdir)

    #overrides
    def get_metric(self, reset       = False):
        u"""
        Returns
        -------
        The average precision, recall and f1.
        """
        recall = self._correct_predicted_brackets / self._gold_brackets if self._gold_brackets > 0 else 0.0
        precision = self._correct_predicted_brackets / self._predicted_brackets if self._gold_brackets > 0 else 0.0
        f1_measure = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        if reset:
            self.reset()
        return {u"evalb_recall": recall, u"evalb_precision": precision, u"evalb_f1_measure": f1_measure}

    #overrides
    def reset(self):
        self._correct_predicted_brackets = 0.0
        self._gold_brackets = 0.0
        self._predicted_brackets = 0.0

    @staticmethod
    def compile_evalb(evalb_directory_path      = DEFAULT_EVALB_DIR):
        os.system(u"cd {} && make && cd ../../../".format(evalb_directory_path))

    @staticmethod
    def clean_evalb(evalb_directory_path      = DEFAULT_EVALB_DIR):
        os.system(u"rm {}".format(os.path.join(evalb_directory_path, u"evalb")))

EvalbBracketingScorer = Metric.register(u"evalb")(EvalbBracketingScorer)
