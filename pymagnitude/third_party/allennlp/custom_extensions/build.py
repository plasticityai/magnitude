# pylint: disable=invalid-name

from __future__ import absolute_import
import os
import torch
from torch.utils.ffi import create_extension

if not torch.cuda.is_available():
    raise Exception(u'HighwayLSTM can only be compiled with CUDA')

sources = [u'src/highway_lstm_cuda.c']
headers = [u'src/highway_lstm_cuda.h']
defines = [(u'WITH_CUDA', None)]
with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = [u'src/highway_lstm_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
        u'_ext.highway_lstm_layer',
        headers=headers,
        sources=sources,
        define_macros=defines,
        relative_to=__file__,
        with_cuda=with_cuda,
        extra_objects=extra_objects
        )

if __name__ == u'__main__':
    ffi.build()
