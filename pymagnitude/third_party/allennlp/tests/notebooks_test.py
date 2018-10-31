from __future__ import with_statement
from __future__ import absolute_import
import os
from io import open

try:
    import nbformat
    from nbconvert.preprocessors.execute import CellExecutionError
    from nbconvert.preprocessors import ExecutePreprocessor
except ModuleNotFoundError:
    print u"jupyter must be installed in order to run notebook tests. "
          u"To install with pip, run: pip install jupyter"

from allennlp.common.testing import AllenNlpTestCase

class TestNotebooks(AllenNlpTestCase):
    def test_vocabulary_tutorial(self):
        assert self.execute_notebook(u"tutorials/notebooks/vocabulary.ipynb")

    def test_data_pipeline_tutorial(self):
        assert self.execute_notebook(u"tutorials/notebooks/data_pipeline.ipynb")

    def test_embedding_tokens_tutorial(self):
        assert self.execute_notebook(u"tutorials/notebooks/embedding_tokens.ipynb")

    @staticmethod
    def execute_notebook(notebook_path     ):
        with open(notebook_path, encoding=u'utf-8') as notebook:
            contents = nbformat.read(notebook, as_version=4)

        execution_processor = ExecutePreprocessor(timeout=60, kernel_name=u"python3")
        try:
            # Actually execute the notebook in the current working directory.
            execution_processor.preprocess(contents, {u'metadata': {u'path': os.getcwdu()}})
            return True
        except CellExecutionError:
            # This is a big chunk of JSON, but the stack trace makes it reasonably
            # clear which cell the error occurred in, so fixing it by actually
            # running the notebook will probably be easier.
            print contents
            return False
