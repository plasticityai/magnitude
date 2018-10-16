u"""
Coreference resolution is defined as follows: given a document, find and cluster entity mentions.
"""


from __future__ import absolute_import
from allennlp.data.dataset_readers.coreference_resolution.conll import ConllCorefReader
from allennlp.data.dataset_readers.coreference_resolution.winobias import WinobiasReader
