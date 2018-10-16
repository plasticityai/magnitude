# pylint: disable=unused-import

from __future__ import absolute_import
import warnings

from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor
warnings.warn(u"allennlp.service.predictors.* has been deprecated."
              u" Please use allennlp.predictors.*", FutureWarning)
