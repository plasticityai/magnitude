# pylint: disable=unused-import

from __future__ import absolute_import
import warnings

from allennlp.predictors.nlvr_parser import NlvrParserPredictor
warnings.warn(u"allennlp.service.predictors.* has been deprecated."
              u" Please use allennlp.predictors.*", FutureWarning)
