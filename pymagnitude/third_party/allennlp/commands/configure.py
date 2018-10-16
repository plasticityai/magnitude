u"""
The ``configure`` subcommand generates a stub configuration for
the specified class (or for the top level configuration if no class specified).

.. code-block:: bash

    $ allennlp configure --help
    usage: allennlp configure [-h] [class]

    Generate a configuration stub for a specific class (or for config as a whole if [class] is omitted).

    positional arguments:
    class

    optional arguments:
    -h, --help            show this help message and exit
"""


from __future__ import absolute_import
from __future__ import print_function
import argparse

from allennlp.commands.subcommand import Subcommand
from allennlp.common.configuration import configure, Config, render_config

class Configure(Subcommand):
    def add_subparser(self, name     , parser                            )                           :
        # pylint: disable=protected-access
        description = u'''Generate a configuration stub for a specific class (or for config as a whole)'''
        subparser = parser.add_parser(
                name, description=description, help=u'Generate configuration stubs.')

        subparser.add_argument(u'cla55', nargs=u'?', default=u'', metavar=u'class')
        subparser.set_defaults(func=_configure)

        return subparser

def _configure(args                    )        :
    cla55 = args.cla55
    parts = cla55.split(u".")
    module = u".".join(parts[:-1])
    class_name = parts[-1]

    print()

    try:
        config = configure(cla55)
        if isinstance(config, Config):
            if cla55:
                print("configuration stub for {cla55}:\n")
            else:
                print("configuration stub for AllenNLP:\n")
            print(render_config(config))
        else:
            print("{class_name} is an abstract base class, choose one of the following subclasses:\n")
            for subclass in config:
                print(u"\t", subclass)
    except ModuleNotFoundError:
        print("unable to load module {module}")
    except AttributeError:
        print("class {class_name} does not exist in module {module}")

    print()
