#!/usr/bin/env python

from __future__ import absolute_import
import logging
import os
import sys

if os.environ.get(u"ALLENNLP_DEBUG"):
    LEVEL = logging.DEBUG
else:
    LEVEL = logging.INFO

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format=u'%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=LEVEL)

from allennlp.commands import main  # pylint: disable=wrong-import-position

if __name__ == u"__main__":
    main(prog=u"allennlp")
