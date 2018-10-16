u"""
The ``test-install`` subcommand verifies
an installation by running the unit tests.

.. code-block:: bash

    $ allennlp test-install --help
    usage: allennlp test-install [-h] [--run-all]
                                 [--include-package INCLUDE_PACKAGE]

    Test that installation works by running the unit tests.

    optional arguments:
      -h, --help            show this help message and exit
      --run-all             By default, we skip tests that are slow or download
                            large files. This flag will run all tests.
      --include-package INCLUDE_PACKAGE
                            additional packages to include
"""


from __future__ import absolute_import
import argparse
import logging
import os
import pathlib

import pytest

import allennlp
from allennlp.commands.subcommand import Subcommand

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class TestInstall(Subcommand):
    def add_subparser(self, name     , parser                            )                           :
        # pylint: disable=protected-access
        description = u'''Test that installation works by running the unit tests.'''
        subparser = parser.add_parser(
                name, description=description, help=u'Run the unit tests.')

        subparser.add_argument(u'--run-all', action=u"store_true",
                               help=u"By default, we skip tests that are slow "
                               u"or download large files. This flag will run all tests.")

        subparser.set_defaults(func=_run_test)

        return subparser


def _get_module_root():
    return pathlib.Path(allennlp.__file__).parent


def _run_test(args                    ):
    initial_working_dir = os.getcwdu()
    module_parent = _get_module_root().parent
    logger.info(u"Changing directory to %s", module_parent)
    os.chdir(module_parent)
    test_dir = os.path.join(module_parent, u"allennlp")
    logger.info(u"Running tests at %s", test_dir)
    if args.run_all:
        # TODO(nfliu): remove this when notebooks have been rewritten as markdown.
        exit_code = pytest.main([test_dir, u'--color=no', u'-k', u'not notebooks_test'])
    else:
        exit_code = pytest.main([test_dir, u'--color=no', u'-k', u'not sniff_test and not notebooks_test',
                                 u'-m', u'not java'])
    # Change back to original working directory after running tests
    os.chdir(initial_working_dir)
    exit(exit_code)
