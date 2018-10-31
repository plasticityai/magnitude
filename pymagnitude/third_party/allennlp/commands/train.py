u"""
The ``train`` subcommand can be used to train a model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ allennlp train --help
   usage: allennlp train [-h] -s SERIALIZATION_DIR
                              [-o OVERRIDES]
                              [--include-package INCLUDE_PACKAGE]
                              [--file-friendly-logging]
                              param_path

   Train the specified model on the specified dataset.

   positional arguments:
   param_path            path to parameter file describing the model to be
                           trained

   optional arguments:
   -h, --help            show this help message and exit
   -s SERIALIZATION_DIR, --serialization-dir SERIALIZATION_DIR
                           directory in which to save the model and its logs
   -o OVERRIDES, --overrides OVERRIDES
                           a JSON structure used to override the experiment
                           configuration
   --include-package INCLUDE_PACKAGE
                           additional packages to include
   --file-friendly-logging
                           outputs tqdm status on separate lines and slows tqdm
                           refresh rate
"""


from __future__ import with_statement
from __future__ import absolute_import
#typing
import argparse
import json
import logging
import os
import re

import torch

from allennlp.commands.evaluate import evaluate
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common import Params
from allennlp.common.util import prepare_environment, prepare_global_logging,\
                                 get_frozen_and_tunable_parameter_names
from allennlp.data import Vocabulary
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.archival import archive_model, CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.trainer import Trainer
from io import open

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Train(Subcommand):
    def add_subparser(self, name     , parser                            )                           :
        # pylint: disable=protected-access
        description = u'''Train the specified model on the specified dataset.'''
        subparser = parser.add_parser(name, description=description, help=u'Train a model')

        subparser.add_argument(u'param_path',
                               type=unicode,
                               help=u'path to parameter file describing the model to be trained')

        subparser.add_argument(u'-s', u'--serialization-dir',
                               required=True,
                               type=unicode,
                               help=u'directory in which to save the model and its logs')

        subparser.add_argument(u'-r', u'--recover',
                               action=u'store_true',
                               default=False,
                               help=u'recover training from the state in serialization_dir')

        subparser.add_argument(u'-o', u'--overrides',
                               type=unicode,
                               default=u"",
                               help=u'a JSON structure used to override the experiment configuration')

        subparser.add_argument(u'--file-friendly-logging',
                               action=u'store_true',
                               default=False,
                               help=u'outputs tqdm status on separate lines and slows tqdm refresh rate')

        subparser.set_defaults(func=train_model_from_args)

        return subparser

def train_model_from_args(args                    ):
    u"""
    Just converts from an ``argparse.Namespace`` object to string paths.
    """
    train_model_from_file(args.param_path,
                          args.serialization_dir,
                          args.overrides,
                          args.file_friendly_logging,
                          args.recover)


def train_model_from_file(parameter_filename     ,
                          serialization_dir     ,
                          overrides      = u"",
                          file_friendly_logging       = False,
                          recover       = False)         :
    u"""
    A wrapper around :func:`train_model` which loads the params from a file.

    Parameters
    ----------
    param_path : ``str``
        A json parameter file specifying an AllenNLP experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs. We just pass this along to
        :func:`train_model`.
    overrides : ``str``
        A JSON string that we will use to override values in the input parameter file.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we make our output more friendly to saved model files.  We just pass this
        along to :func:`train_model`.
    recover : ``bool`, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.
    """
    # Load the experiment config from a file and pass it to ``train_model``.
    params = Params.from_file(parameter_filename, overrides)
    return train_model(params, serialization_dir, file_friendly_logging, recover)


def datasets_from_params(params        )                                 :
    u"""
    Load all the datasets specified by the config.
    """
    dataset_reader = DatasetReader.from_params(params.pop(u'dataset_reader'))
    validation_dataset_reader_params = params.pop(u"validation_dataset_reader", None)

    validation_and_test_dataset_reader                = dataset_reader
    if validation_dataset_reader_params is not None:
        logger.info(u"Using a separate dataset reader to load validation and test data.")
        validation_and_test_dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

    train_data_path = params.pop(u'train_data_path')
    logger.info(u"Reading training data from %s", train_data_path)
    train_data = dataset_reader.read(train_data_path)

    datasets                                = {u"train": train_data}

    validation_data_path = params.pop(u'validation_data_path', None)
    if validation_data_path is not None:
        logger.info(u"Reading validation data from %s", validation_data_path)
        validation_data = validation_and_test_dataset_reader.read(validation_data_path)
        datasets[u"validation"] = validation_data

    test_data_path = params.pop(u"test_data_path", None)
    if test_data_path is not None:
        logger.info(u"Reading test data from %s", test_data_path)
        test_data = validation_and_test_dataset_reader.read(test_data_path)
        datasets[u"test"] = test_data

    return datasets

def create_serialization_dir(params        , serialization_dir     , recover      )        :
    u"""
    This function creates the serialization directory if it doesn't exist.  If it already exists
    and is non-empty, then it verifies that we're recovering from a training with an identical configuration.

    Parameters
    ----------
    params: ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir: ``str``
        The directory in which to save results and logs.
    recover: ``bool``
        If ``True``, we will try to recover from an existing serialization directory, and crash if
        the directory doesn't exist, or doesn't match the configuration we're given.
    """
    if os.path.exists(serialization_dir) and os.listdir(serialization_dir):
        if not recover:
            raise ConfigurationError("Serialization directory ({serialization_dir}) already exists and is "
                                     "not empty. Specify --recover to recover training from existing output.")

        logger.info("Recovering from prior training at {serialization_dir}.")

        recovered_config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(recovered_config_file):
            raise ConfigurationError(u"The serialization directory already exists but doesn't "
                                     u"contain a config.json. You probably gave the wrong directory.")
        else:
            loaded_params = Params.from_file(recovered_config_file)

            # Check whether any of the training configuration differs from the configuration we are
            # resuming.  If so, warn the user that training may fail.
            fail = False
            flat_params = params.as_flat_dict()
            flat_loaded = loaded_params.as_flat_dict()
            for key in list(flat_params.keys()) - list(flat_loaded.keys()):
                logger.error("Key '{key}' found in training configuration but not in the serialization "
                             "directory we're recovering from.")
                fail = True
            for key in list(flat_loaded.keys()) - list(flat_params.keys()):
                logger.error("Key '{key}' found in the serialization directory we're recovering from "
                             "but not in the training config.")
                fail = True
            for key in list(flat_params.keys()):
                if flat_params.get(key, None) != flat_loaded.get(key, None):
                    logger.error("Value for '{key}' in training configuration does not match that the value in "
                                 "the serialization directory we're recovering from: "
                                 "{flat_params[key]} != {flat_loaded[key]}")
                    fail = True
            if fail:
                raise ConfigurationError(u"Training configuration does not match the configuration we're "
                                         u"recovering from.")
    else:
        if recover:
            raise ConfigurationError("--recover specified but serialization_dir ({serialization_dir}) "
                                     u"does not exist.  There is nothing to recover from.")
        os.makedirs(serialization_dir, exist_ok=True)


def train_model(params        ,
                serialization_dir     ,
                file_friendly_logging       = False,
                recover       = False)         :
    u"""
    Trains the model specified in the given :class:`Params` object, using the data and training
    parameters also specified in that object, and saves the results in ``serialization_dir``.

    Parameters
    ----------
    params : ``Params``
        A parameter object specifying an AllenNLP Experiment.
    serialization_dir : ``str``
        The directory in which to save results and logs.
    file_friendly_logging : ``bool``, optional (default=False)
        If ``True``, we add newlines to tqdm output, even on an interactive terminal, and we slow
        down tqdm's output to only once every 10 seconds.
    recover : ``bool``, optional (default=False)
        If ``True``, we will try to recover a training run from an existing serialization
        directory.  This is only intended for use when something actually crashed during the middle
        of a run.  For continuing training a model on new data, see the ``fine-tune`` command.

    Returns
    -------
    best_model: ``Model``
        The model with the best epoch weights.
    """
    prepare_environment(params)

    create_serialization_dir(params, serialization_dir, recover)
    prepare_global_logging(serialization_dir, file_friendly_logging)

    check_for_gpu(params.get(u'trainer').get(u'cuda_device', -1))

    params.to_file(os.path.join(serialization_dir, CONFIG_NAME))

    all_datasets = datasets_from_params(params)
    datasets_for_vocab_creation = set(params.pop(u"datasets_for_vocab_creation", all_datasets))

    for dataset in datasets_for_vocab_creation:
        if dataset not in all_datasets:
            raise ConfigurationError("invalid 'dataset_for_vocab_creation' {dataset}")

    logger.info(u"From dataset instances, %s will be considered for vocabulary creation.",
                u", ".join(datasets_for_vocab_creation))
    vocab = Vocabulary.from_params(
            params.pop(u"vocabulary", {}),
            (instance for key, dataset in list(all_datasets.items())
             for instance in dataset
             if key in datasets_for_vocab_creation)
    )

    vocab.save_to_files(os.path.join(serialization_dir, u"vocabulary"))

    model = Model.from_params(vocab=vocab, params=params.pop(u'model'))
    iterator = DataIterator.from_params(params.pop(u"iterator"))
    iterator.index_with(vocab)
    validation_iterator_params = params.pop(u"validation_iterator", None)
    if validation_iterator_params:
        validation_iterator = DataIterator.from_params(validation_iterator_params)
        validation_iterator.index_with(vocab)
    else:
        validation_iterator = None

    train_data = all_datasets[u'train']
    validation_data = all_datasets.get(u'validation')
    test_data = all_datasets.get(u'test')

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

    trainer = Trainer.from_params(model,
                                  serialization_dir,
                                  iterator,
                                  train_data,
                                  validation_data,
                                  trainer_params,
                                  validation_iterator=validation_iterator)

    evaluate_on_test = params.pop_bool(u"evaluate_on_test", False)
    params.assert_empty(u'base train command')

    try:
        metrics = trainer.train()
    except KeyboardInterrupt:
        # if we have completed an epoch, try to create a model archive.
        if os.path.exists(os.path.join(serialization_dir, _DEFAULT_WEIGHTS)):
            logging.info(u"Training interrupted by the user. Attempting to create "
                         u"a model archive using the current best epoch weights.")
            archive_model(serialization_dir, files_to_archive=params.files_to_archive)
        raise

    # Now tar up results
    archive_model(serialization_dir, files_to_archive=params.files_to_archive)

    logger.info(u"Loading the best epoch weights.")
    best_model_state_path = os.path.join(serialization_dir, u'best.th')
    best_model_state = torch.load(best_model_state_path)
    best_model = model
    best_model.load_state_dict(best_model_state)

    if test_data and evaluate_on_test:
        logger.info(u"The model will be evaluated using the best epoch weights.")
        test_metrics = evaluate(
                best_model, test_data, validation_iterator or iterator,
                cuda_device=trainer._cuda_devices[0] # pylint: disable=protected-access
        )
        for key, value in list(test_metrics.items()):
            metrics[u"test_" + key] = value

    elif test_data:
        logger.info(u"To evaluate on the test set after training, pass the "
                    u"'evaluate_on_test' flag, or use the 'allennlp evaluate' command.")

    metrics_json = json.dumps(metrics, indent=2)
    with open(os.path.join(serialization_dir, u"metrics.json"), u"w") as metrics_file:
        metrics_file.write(metrics_json)
    logger.info(u"Metrics: %s", metrics_json)

    return best_model
