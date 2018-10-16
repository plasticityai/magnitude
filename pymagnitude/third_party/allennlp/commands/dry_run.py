u"""
The ``dry-run`` command creates a vocabulary, informs you of
dataset statistics and other training utilities without actually training
a model.

.. code-block:: bash

    $ allennlp dry-run --help
    usage: allennlp dry-run [-h] -s SERIALIZATION_DIR [-o OVERRIDES]
                                      [--include-package INCLUDE_PACKAGE]
                                      param_path

    Create a vocabulary, compute dataset statistics and other training utilities.

    positional arguments:
    param_path            path to parameter file describing the model and its
                            inputs

    optional arguments:
    -h, --help            show this help message and exit
    -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                            directory in which to save the output of the dry run.
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
import re

from allennlp.commands.train import datasets_from_params
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.params import Params
from allennlp.common.util import prepare_environment, get_frozen_and_tunable_parameter_names
from allennlp.data import Vocabulary
from allennlp.data.dataset import Batch
from allennlp.models import Model

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DryRun(Subcommand):
    def add_subparser(self, name     , parser                            )                           :
        # pylint: disable=protected-access
        description = u'''Create a vocabulary, compute dataset statistics and other training utilities.'''
        subparser = parser.add_parser(name,
                                      description=description,
                                      help=u'Create a vocabulary, compute dataset statistics '
                                           u'and other training utilities.')
        subparser.add_argument(u'param_path',
                               type=unicode,
                               help=u'path to parameter file describing the model and its inputs')
        subparser.add_argument(u'-s', u'--serialization-dir',
                               required=True,
                               type=unicode,
                               help=u'directory in which to save the output of the dry run.')

        subparser.add_argument(u'-o', u'--overrides',
                               type=unicode,
                               default=u"",
                               help=u'a JSON structure used to override the experiment configuration')

        subparser.set_defaults(func=dry_run_from_args)

        return subparser


def dry_run_from_args(args                    ):
    u"""
    Just converts from an ``argparse.Namespace`` object to params.
    """
    parameter_path = args.param_path
    serialization_dir = args.serialization_dir
    overrides = args.overrides

    params = Params.from_file(parameter_path, overrides)

    dry_run_from_params(params, serialization_dir)

def dry_run_from_params(params        , serialization_dir     )        :
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
    dataset = Batch(instances)
    dataset.index_instances(vocab)
    dataset.print_statistics()
    vocab.print_statistics()

    logger.info("writing the vocabulary to {vocab_dir}.")
    vocab.save_to_files(vocab_dir)

    model = Model.from_params(vocab=vocab, params=params.pop(u'model'))
    trainer_params = params.pop(u"trainer")
    no_grad_regexes = trainer_params.pop(u"no_grad", ())
    for name, parameter in model.named_parameters():
        if any(re.search(regex, name) for regex in no_grad_regexes):
            parameter.requires_grad_(False)

    frozen_parameter_names, tunable_parameter_names =\
                   get_frozen_and_tunable_parameter_names(model)
    logger.info(u"Following parameters are Frozen  (without gradient):")
    for name in frozen_parameter_names:
        logger.info(name)
    logger.info(u"Following parameters are Tunable (with gradient):")
    for name in tunable_parameter_names:
        logger.info(name)
