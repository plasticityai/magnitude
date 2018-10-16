u"""
The ``predict`` subcommand allows you to make bulk JSON-to-JSON
or dataset to JSON predictions using a trained model and its
:class:`~allennlp.service.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ allennlp predict -h
    usage: allennlp predict [-h] [--output-file OUTPUT_FILE]
                            [--weights-file WEIGHTS_FILE]
                            [--batch-size BATCH_SIZE] [--silent]
                            [--cuda-device CUDA_DEVICE] [--use-dataset-reader]
                            [-o OVERRIDES] [--predictor PREDICTOR]
                            [--include-package INCLUDE_PACKAGE]
                            archive_file input_file

    Run the specified model against a JSON-lines input file.

    positional arguments:
    archive_file          the archived model to make predictions with
    input_file            path to input file

    optional arguments:
    -h, --help              show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file
    --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
    --batch-size BATCH_SIZE The batch size to use for processing
    --silent                do not print output to stdout
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    --use-dataset-reader    Whether to use the dataset reader of the original
                            model to load Instances
    -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
    --predictor PREDICTOR   optionally specify a specific predictor to use
    --include-package INCLUDE_PACKAGE
                            additional packages to include
"""

from __future__ import absolute_import
from __future__ import print_function
#typing
import argparse
import sys
import json

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance
from io import open
try:
    from itertools import izip
except:
    izip = zip


class Predict(Subcommand):
    def add_subparser(self, name     , parser                            )                           :
        # pylint: disable=protected-access
        description = u'''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help=u'Use a trained model to make predictions.')

        subparser.add_argument(u'archive_file', type=unicode, help=u'the archived model to make predictions with')
        subparser.add_argument(u'input_file', type=unicode, help=u'path to input file')

        subparser.add_argument(u'--output-file', type=unicode, help=u'path to output file')
        subparser.add_argument(u'--weights-file',
                               type=unicode,
                               help=u'a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument(u'--batch-size', type=int, default=1, help=u'The batch size to use for processing')

        subparser.add_argument(u'--silent', action=u'store_true', help=u'do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument(u'--cuda-device', type=int, default=-1, help=u'id of GPU to use (if any)')

        subparser.add_argument(u'--use-dataset-reader',
                               action=u'store_true',
                               help=u'Whether to use the dataset reader of the original model to load Instances')

        subparser.add_argument(u'-o', u'--overrides',
                               type=unicode,
                               default=u"",
                               help=u'a JSON structure used to override the experiment configuration')

        subparser.add_argument(u'--predictor',
                               type=unicode,
                               help=u'optionally specify a specific predictor to use')

        subparser.set_defaults(func=_predict)

        return subparser

def _get_predictor(args                    )             :
    check_for_gpu(args.cuda_device)
    archive = load_archive(args.archive_file,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)

    return Predictor.from_archive(archive, args.predictor)


class _PredictManager(object):

    def __init__(self,
                 predictor           ,
                 input_file     ,
                 output_file               ,
                 batch_size     ,
                 print_to_console      ,
                 has_dataset_reader      )        :

        self._predictor = predictor
        self._input_file = input_file
        if output_file is not None:
            self._output_file = open(output_file, u"w")
        else:
            self._output_file = None
        self._batch_size = batch_size
        self._print_to_console = print_to_console
        if has_dataset_reader:
            self._dataset_reader = predictor._dataset_reader # pylint: disable=protected-access
        else:
            self._dataset_reader = None

    def _predict_json(self, batch_data                )                 :
        if len(batch_data) == 1:
            results = [self._predictor.predict_json(batch_data[0])]
        else:
            results = self._predictor.predict_batch_json(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _predict_instances(self, batch_data                )                 :
        if len(batch_data) == 1:
            results = [self._predictor.predict_instance(batch_data[0])]
        else:
            results = self._predictor.predict_batch_instance(batch_data)
        for output in results:
            yield self._predictor.dump_line(output)

    def _maybe_print_to_console_and_file(self,
                                         prediction     ,
                                         model_input      = None)        :
        if self._print_to_console:
            if model_input is not None:
                print(u"input: ", model_input)
            print(u"prediction: ", prediction)
        if self._output_file is not None:
            self._output_file.write(prediction)

    def _get_json_data(self)                      :
        for line in open(self._input_file):
            if not line.isspace():
                yield self._predictor.load_line(line)

    def _get_instance_data(self)                      :
        if self._dataset_reader is None:
            raise ConfigurationError(u"To generate instances directly, pass a DatasetReader.")
        else:
            yield self._dataset_reader.read(self._input_file)

    def run(self)        :
        has_reader = self._dataset_reader is not None
        if has_reader:
            for batch in lazy_groups_of(self._get_instance_data(), self._batch_size):
                for result in self._predict_instances(batch):
                    self._maybe_print_to_console_and_file(result)
        else:
            for batch_json in lazy_groups_of(self._get_json_data(), self._batch_size):
                for model_input, result in izip(batch_json, self._predict_json(batch_json)):
                    self._maybe_print_to_console_and_file(result, json.dumps(model_input))

        if self._output_file is not None:
            self._output_file.close()

def _predict(args                    )        :
    predictor = _get_predictor(args)

    if args.silent and not args.output_file:
        print(u"--silent specified without --output-file.")
        print(u"Exiting early because no output will be created.")
        sys.exit(0)

    manager = _PredictManager(predictor,
                              args.input_file,
                              args.output_file,
                              args.batch_size,
                              not args.silent,
                              args.use_dataset_reader)
    manager.run()
