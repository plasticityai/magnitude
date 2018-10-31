u"""
Base class for subcommands under ``allennlp.run``.
"""

from __future__ import absolute_import
import argparse

class Subcommand(object):
    u"""
    An abstract class representing subcommands for allennlp.run.
    If you wanted to (for example) create your own custom `special-evaluate` command to use like

    ``allennlp special-evaluate ...``

    you would create a ``Subcommand`` subclass and then pass it as an override to
    :func:`~allennlp.commands.main` .
    """
    def add_subparser(self, name     , parser                            )                           :
        # pylint: disable=protected-access
        raise NotImplementedError
