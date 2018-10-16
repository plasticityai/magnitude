# pylint: disable=access-member-before-definition

from __future__ import absolute_import
#typing

#overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField


class SpanField(Field):
    u"""
    A ``SpanField`` is a pair of inclusive, zero-indexed (start, end) indices into a
    :class:`~allennlp.data.fields.sequence_field.SequenceField`, used to represent a span of text.
    Because it's a pair of indices into a :class:`SequenceField`, we take one of those as input
    to make the span's dependence explicit and to validate that the span is well defined.

    Parameters
    ----------
    span_start : ``int``, required.
        The index of the start of the span in the :class:`SequenceField`.
    span_end : ``int``, required.
        The inclusive index of the end of the span in the :class:`SequenceField`.
    sequence_field : ``SequenceField``, required.
        A field containing the sequence that this ``SpanField`` is a span inside.
    """
    def __init__(self, span_start     , span_end     , sequence_field               )        :
        self.span_start = span_start
        self.span_end = span_end
        self.sequence_field = sequence_field

        if not isinstance(span_start, int) or not isinstance(span_end, int):
            raise TypeError("SpanFields must be passed integer indices. Found span indices: "
                            "({span_start}, {span_end}) with types "
                            "({type(span_start)} {type(span_end)})")
        if span_start > span_end:
            raise ValueError("span_start must be less than span_end, "
                             "but found ({span_start}, {span_end}).")

        if span_end > self.sequence_field.sequence_length() - 1:
            raise ValueError("span_end must be < len(sequence_length) - 1, but found "
                             "{span_end} and {self.sequence_field.sequence_length() - 1} respectively.")

    #overrides
    def get_padding_lengths(self)                  :
        # pylint: disable=no-self-use
        return {}

    #overrides
    def as_tensor(self,
                  padding_lengths                ,
                  cuda_device      = -1)                :
        # pylint: disable=unused-argument
        tensor = torch.LongTensor([self.span_start, self.span_end])
        return tensor if cuda_device == -1 else tensor.cuda(cuda_device)

    #overrides
    def empty_field(self):
        return SpanField(-1, -1, self.sequence_field.empty_field())

    def __str__(self)       :
        return "SpanField with spans: ({self.span_start}, {self.span_end})."
