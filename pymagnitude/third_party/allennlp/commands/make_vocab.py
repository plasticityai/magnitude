u"""
The ``make-vocab`` subcommand allows you to create a vocabulary from
your dataset[s], which you can then reuse without recomputing it
each training run.

.. code-block:: bash

   $ allennlp make-vocab --help

    usage: allennlp make-vocab [-h] [-o OVERRIDES] [--include-package INCLUDE_PACKAGE] param_path

    Create a vocabulary from the specified dataset.

    positional arguments:
    param_path            path to parameter file describing the model and its
                          inputs

    optional arguments:
    -h, --help            show this help message and exit
   -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                           directory in which to save the vocabulary directory
    -o OVERRIDES, --overrides OVERRIDES
                          a JSON structure used to override the experiment
                          configuration
    --include-package INCLUDE_PACKAGE
                            additional packages to include
"""

from __future__ import absolute_import
import argparse
import logging
import os

from allennlp.commands.train import datasets_from_params
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment
from allennlp.data import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class MakeVocab(Subcommand):
    def add_subparser(self, name     , parser                            )                           :
        # pylint: disable=protected-access
        description = u'''Create a vocabulary from the specified dataset.'''
        subparser = parser.add_parser(
                name, description=description, help=u'Create a vocabulary')
        subparser.add_argument(u'param_path',
                               type=unicode,
                               help=u'path to parameter file describing the model and its inputs')

        subparser.add_argument(u'-s', u'--serialization-dir',
                               required=True,
                               type=unicode,
                               help=u'directory in which to save the vocabulary directory')

        subparser.add_argument(u'-o', u'--overrides',
                               type=unicode,
                               default=u"",
                               help=u'a JSON structure used to override the experiment configuration')

        subparser.set_defaults(func=make_vocab_from_args)

        return subparser


def make_vocab_from_args(args                    ):
    u"""
    Just converts from an ``argparse.Namespace`` object to params.
    """
    parameter_path = args.param_path
    overrides = args.overrides
    serialization_dir = args.serialization_dir

    params = Params.from_file(parameter_path, overrides)

    make_vocab_from_params(params, serialization_dir)

def make_vocab_from_params(params        , serialization_dir     ):
    prepare_environment(params)

    vocab_params = params.pop(u"vocabulary", {})
    os.makedirs(serialization_dir, exist_ok=True)
    vocab_dir = os.path.join(serialization_dir, u"vocabulary")

    if os.path.isdir(vocab_dir) and os.listdir(vocab_dir) is not None:
        raise ConfigurationError(u"The 'vocabulary' directory in the provided "
                                 u"serialization directory is non-empty")

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop(u"datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError("invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info(u"From dataset instances, %s will be considered for vocabulary creation.",
                u", ".join(datasets_for_vocab_creation))

    instances = [instance for key, dataset in list(all_datasets.items())
                 for instance in dataset
                 if key in datasets_for_vocab_creation]

    vocab = Vocabulary.from_params(vocab_params, instances)

    logger.info("writing the vocabulary to {vocab_dir}.")
    vocab.save_to_files(vocab_dir)
    logger.info(u"done creating vocab")
