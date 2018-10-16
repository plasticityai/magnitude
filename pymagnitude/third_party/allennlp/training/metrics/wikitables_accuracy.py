

from __future__ import division
from __future__ import absolute_import
import atexit
import logging
import os
import pathlib
import subprocess

#overrides

from allennlp.common.file_utils import cached_path
from allennlp.training.metrics.metric import Metric

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

SEMPRE_EXECUTOR_JAR = u"https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-executor-0.1.0.jar"
ABBREVIATIONS_FILE = u"https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-abbreviations.tsv"
GROW_FILE = u"https://s3-us-west-2.amazonaws.com/allennlp/misc/wikitables-grow.grammar"
SEMPRE_DIR = unicode(pathlib.Path(u'data/'))
SEMPRE_ABBREVIATIONS_PATH = os.path.join(SEMPRE_DIR, u"abbreviations.tsv")
SEMPRE_GRAMMAR_PATH = os.path.join(SEMPRE_DIR, u"grow.grammar")


class WikiTablesAccuracy(Metric):
    def __init__(self, table_directory     )        :
        self._table_directory = table_directory
        self._executor_process: subprocess.Popen = None
        self._create_sempre_executor()
        self._count = 0
        self._correct = 0

    #overrides
    def __call__(self, logical_form     , example_lisp_string     ):  # type: ignore
        u"""
        Parameters
        ----------
        example_lisp_string : ``str``
            The value to average.
        """
        denotation_correct = self.evaluate_logical_form(logical_form, example_lisp_string)
        if denotation_correct:
            self._correct += 1
        self._count += 1

    #overrides
    def get_metric(self, reset       = False)         :
        accuracy = self._correct / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return accuracy

    #overrides
    def reset(self):
        self._count = 0
        self._correct = 0

    def __str__(self):
        return "WikiTablesAccuracy(correct={self._correct}, count={self._count})"

    def evaluate_logical_form(self, logical_form     , example_lisp_string     )        :
        if not logical_form or logical_form.startswith(u'Error'):
            return False
        if example_lisp_string[-1] != u'\n':
            example_lisp_string += u'\n'
        if logical_form[-1] != u'\n':
            logical_form += u'\n'
        self._executor_process.stdin.write(example_lisp_string.encode(u'utf-8'))
        self._executor_process.stdin.write(logical_form.encode(u'utf-8'))
        self._executor_process.stdin.flush()
        result = self._executor_process.stdout.readline().decode().strip()
        return result == u'1.0'

    def _create_sempre_executor(self)        :
        u"""
        Creates a server running SEMPRE that we can send logical forms to for evaluation.  This
        uses inter-process communication, because SEMPRE is java code.  We also need to be careful
        to clean up the process when our program exits.
        """
        if self._executor_process:
            return

        # It'd be much nicer to just use `cached_path` for these files.  However, the SEMPRE jar
        # that we're using expects to find these files in a particular location, so we need to make
        # sure we put the files in that location.
        os.makedirs(SEMPRE_DIR, exist_ok=True)
        abbreviations_path = os.path.join(SEMPRE_DIR, u'abbreviations.tsv')
        if not os.path.exists(abbreviations_path):
            subprocess.run('wget {ABBREVIATIONS_FILE}', shell=True)
            subprocess.run('mv wikitables-abbreviations.tsv {abbreviations_path}', shell=True)

        grammar_path = os.path.join(SEMPRE_DIR, u'grow.grammar')
        if not os.path.exists(grammar_path):
            subprocess.run('wget {GROW_FILE}', shell=True)
            subprocess.run('mv wikitables-grow.grammar {grammar_path}', shell=True)

        args = [u'java', u'-jar', cached_path(SEMPRE_EXECUTOR_JAR), u'serve', self._table_directory]
        self._executor_process = subprocess.Popen(args,
                                                  stdin=subprocess.PIPE,
                                                  stdout=subprocess.PIPE,
                                                  bufsize=1)

        lines = []
        for _ in range(6):
            # SEMPRE outputs six lines of stuff when it loads that I can't disable.  So, we clear
            # that here.
            lines.append(unicode(self._executor_process.stdout.readline()))
        assert u'Parser' in lines[-1], u"SEMPRE server output unexpected; the server may have changed"
        logger.info(u"Started SEMPRE server for evaluating logical forms")

        # This is supposed to ensure that the subprocess gets killed when python exits.
        atexit.register(self._stop_sempre_executor)

    def _stop_sempre_executor(self)        :
        if not self._executor_process:
            return
        self._executor_process.terminate()
        self._executor_process = None
        logger.info(u"Stopped SEMPRE server")
