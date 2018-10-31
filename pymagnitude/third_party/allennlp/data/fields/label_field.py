
from __future__ import absolute_import
#typing
import logging

#overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class LabelField(Field):
    u"""
    A ``LabelField`` is a categorical label of some kind, where the labels are either strings of
    text or 0-indexed integers (if you wish to skip indexing by passing skip_indexing=True).
    If the labels need indexing, we will use a :class:`Vocabulary` to convert the string labels
    into integers.

    This field will get converted into an integer index representing the class label.

    Parameters
    ----------
    label : ``Union[str, int]``
    label_namespace : ``str``, optional (default="labels")
        The namespace to use for converting label strings into integers.  We map label strings to
        integers for you (e.g., "entailment" and "contradiction" get converted to 0, 1, ...),
        and this namespace tells the ``Vocabulary`` object which mapping from strings to integers
        to use (so "entailment" as a label doesn't get the same integer id as "entailment" as a
        word).  If you have multiple different label fields in your data, you should make sure you
        use different namespaces for each one, always using the suffix "labels" (e.g.,
        "passage_labels" and "question_labels").
    skip_indexing : ``bool``, optional (default=False)
        If your labels are 0-indexed integers, you can pass in this flag, and we'll skip the indexing
        step.  If this is ``False`` and your labels are not strings, this throws a ``ConfigurationError``.
    """
    # Most often, you probably don't want to have OOV/PAD tokens with a LabelField, so we warn you
    # about it when you pick a namespace that will getting these tokens by default.  It is
    # possible, however, that you _do_ actually want OOV/PAD tokens with this Field.  This class
    # variable is used to make sure that we only log a single warning for this per namespace, and
    # not every time you create one of these Field objects.
    _already_warned_namespaces           = set()

    def __init__(self,
                 label                 ,
                 label_namespace      = u'labels',
                 skip_indexing       = False)        :
        self.label = label
        self._label_namespace = label_namespace
        self._label_id = None
        self._maybe_warn_for_namespace(label_namespace)

        if skip_indexing:
            if not isinstance(label, int):
                raise ConfigurationError(u"In order to skip indexing, your labels must be integers. "
                                         u"Found label = {}".format(label))
            else:
                self._label_id = label
        else:
            if not isinstance(label, unicode):
                raise ConfigurationError(u"LabelFields must be passed a string label if skip_indexing=False. "
                                         u"Found label: {} with type: {}.".format(label, type(label)))

    def _maybe_warn_for_namespace(self, label_namespace     )        :
        if not (self._label_namespace.endswith(u"labels") or self._label_namespace.endswith(u"tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning(u"Your label namespace was '%s'. We recommend you use a namespace "
                               u"ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                               u"default to your vocabulary.  See documentation for "
                               u"`non_padded_namespaces` parameter in Vocabulary.",
                               self._label_namespace)
                self._already_warned_namespaces.add(label_namespace)

    #overrides
    def count_vocab_items(self, counter                           ):
        if self._label_id is None:
            counter[self._label_namespace][self.label] += 1  # type: ignore

    #overrides
    def index(self, vocab            ):
        if self._label_id is None:
            self._label_id = vocab.get_token_index(self.label, self._label_namespace)  # type: ignore

    #overrides
    def get_padding_lengths(self)                  :  # pylint: disable=no-self-use
        return {}

    #overrides
    def as_tensor(self,
                  padding_lengths                ,
                  cuda_device      = -1)                :
        # pylint: disable=unused-argument,not-callable
        tensor = torch.tensor(self._label_id, dtype=torch.long)
        return tensor if cuda_device == -1 else tensor.cuda(cuda_device)

    #overrides
    def empty_field(self):
        return LabelField(-1, self._label_namespace, skip_indexing=True)

    def __str__(self)       :
        return "LabelField with label: {self.label} in namespace: '{self._label_namespace}'.'"
