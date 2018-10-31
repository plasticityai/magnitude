
from __future__ import absolute_import
#typing
import argparse
import logging


from allennlp.commands.configure import Configure
from allennlp.commands.elmo import Elmo
from allennlp.commands.evaluate import Evaluate
from allennlp.commands.fine_tune import FineTune
from allennlp.commands.make_vocab import MakeVocab
from allennlp.commands.predict import Predict
from allennlp.commands.dry_run import DryRun
from allennlp.commands.subcommand import Subcommand
from allennlp.commands.test_install import TestInstall
from allennlp.commands.train import Train
from allennlp.common.util import import_submodules

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main(prog      = None,
         subcommand_overrides                        = {})        :
    u"""
    The :mod:`~allennlp.run` command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own ``Model`` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag.
    """
    # pylint: disable=dangerous-default-value
    parser = argparse.ArgumentParser(description=u"Run AllenNLP", usage=u'%(prog)s', prog=prog)
    parser.add_argument(u'--version', action=u'version', version=u'%(prog)s ')

    subparsers = parser.add_subparsers(title=u'Commands', metavar=u'')

    subcommands = {
            # Default commands
            u"configure": Configure(),
            u"train": Train(),
            u"evaluate": Evaluate(),
            u"predict": Predict(),
            u"make-vocab": MakeVocab(),
            u"elmo": Elmo(),
            u"fine-tune": FineTune(),
            u"dry-run": DryRun(),
            u"test-install": TestInstall(),

  
    }

    for name, subcommand in list(subcommands.items()):
        subparser = subcommand.add_subparser(name, subparsers)
        # configure doesn't need include-package because it imports
        # whatever classes it needs.
        if name != u"configure":
            subparser.add_argument(u'--include-package',
                                   type=unicode,
                                   action=u'append',
                                   default=[],
                                   help=u'additional packages to include')

    args = parser.parse_args()

    # If a subparser is triggered, it adds its work as `args.func`.
    # So if no such attribute has been added, no subparser was triggered,
    # so give the user some help.
    if u'func' in dir(args):
        # Import any additional modules needed (to register custom classes).
        for package_name in getattr(args, u'include_package', ()):
            import_submodules(package_name)
        args.func(args)
    else:
        parser.print_help()
