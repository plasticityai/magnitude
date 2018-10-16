# pylint: disable=invalid-name,no-self-use



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import os
import sys
from io import StringIO

from allennlp.commands import main
from allennlp.common.testing import AllenNlpTestCase
from io import open


class TestConfigure(AllenNlpTestCase):

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / u'configuretestpackage'
        packagedir.mkdir()  # pylint: disable=no-member
        (packagedir / u'__init__.py').touch()  # pylint: disable=no-member

        # And add that directory to the path
        sys.path.insert(0, unicode(self.TEST_DIR))

        # Write out a duplicate predictor there, but registered under a different name.
        from allennlp.predictors import bidaf
        with open(bidaf.__file__) as f:
            code = f.read().replace(u"""@Predictor.register('machine-comprehension')""",
                                    u"""@Predictor.register('configure-test-predictor')""")

        with open(os.path.join(packagedir, u'predictor.py'), u'w') as f:
            f.write(code)

        # Capture stdout
        stdout_saved = sys.stdout
        stdout_captured = StringIO()
        sys.stdout = stdout_captured

        sys.argv = [u"run.py",      # executable
                    u"configure",     # command
                    u"configuretestpackage.predictor.BidafPredictor"]

        main()
        output = stdout_captured.getvalue()
        assert u"configure-test-predictor" in output

        sys.stdout = stdout_saved

        sys.path.remove(unicode(self.TEST_DIR))
