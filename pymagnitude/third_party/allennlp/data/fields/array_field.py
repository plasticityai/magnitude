
from __future__ import absolute_import
#typing

import numpy
import torch
#overrides

from allennlp.data.fields.field import Field


class ArrayField(Field):
    u"""
    A class representing an array, which could have arbitrary dimensions.
    A batch of these arrays are padded to the max dimension length in the batch
    for each dimension.
    """
    def __init__(self, array               , padding_value      = 0)        :
        self.array = array
        self.padding_value = padding_value

    #overrides
    def get_padding_lengths(self)                  :
        return dict((u"dimension_" + unicode(i), shape)
                for i, shape in enumerate(self.array.shape))

    #overrides
    def as_tensor(self,
                  padding_lengths                ,
                  cuda_device      = -1)                :
        max_shape = [padding_lengths[u"dimension_{}".format(i)]
                     for i in range(len(padding_lengths))]

        return_array = numpy.ones(max_shape, u"float32") * self.padding_value

        # If the tensor has a different shape from the largest tensor, pad dimensions with zeros to
        # form the right shaped list of slices for insertion into the final tensor.
        slicing_shape = list(self.array.shape)
        if len(self.array.shape) < len(max_shape):
            slicing_shape = slicing_shape + [0 for _ in range(len(max_shape) - len(self.array.shape))]
        slices = tuple([slice(0, x) for x in slicing_shape])
        return_array[slices] = self.array
        tensor = torch.from_numpy(return_array)
        return tensor if cuda_device == -1 else tensor.cuda(cuda_device)

    #overrides
    def empty_field(self):  # pylint: disable=no-self-use
        # Pass the padding_value, so that any outer field, e.g., `ListField[ArrayField]` uses the
        # same padding_value in the padded ArrayFields
        return ArrayField(numpy.array([], dtype=u"float32"), padding_value=self.padding_value)


    def __str__(self)       :
        return "ArrayField with shape: {self.array.shape}."
