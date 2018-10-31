# pylint: disable=no-self-use,invalid-name



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import json
import tarfile

import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token
from allennlp.data.token_indexers import OpenaiTransformerBytePairIndexer
from io import open


class TestOpenaiTransformerBytePairIndexer(AllenNlpTestCase):
    def setUp(self):
        super(TestOpenaiTransformerBytePairIndexer, self).setUp()

        encoder_path = self.TEST_DIR / u'encoder.json'
        bpe_path = self.TEST_DIR / u'vocab.bpe'
        transformer_model_path = self.TEST_DIR / u'model.tar.gz'

        symbols = [u"e", u"w", u"o", u"wo", u"."]
        byte_pairs = [(sym1, sym2 + end)
                      for sym1 in symbols        # prefer earlier first symbol
                      for sym2 in symbols        # if tie, prefer earlier second symbol
                      for end in (u'</w>', u'')]   # if tie, prefer ending a word
        encoding = dict(("{sym1}{sym2}", idx + 1) for idx, (sym1, sym2) in enumerate(byte_pairs))


        with open(encoder_path, u'w') as encoder_file:
            json.dump(encoding, encoder_file)

        with open(bpe_path, u'w') as bpe_file:
            bpe_file.write(u"#version 0.0\n")
            for sym1, sym2 in byte_pairs:
                bpe_file.write("{sym1} {sym2}\n")
            bpe_file.write(u"\n")

        with tarfile.open(transformer_model_path, u'w') as tf:
            tf.add(encoder_path, u'model/encoder_bpe_40000.json')
            tf.add(bpe_path, u'model/vocab_40000.bpe')

        self.indexer = OpenaiTransformerBytePairIndexer(encoding, byte_pairs)

    def test_bpe(self):

        # [e, w, o, e</w>] -> best pair (e, w)
        # [ew, o, e</w>] -> best pair (o, e</w>)
        # [ew, oe</w>] -> done
        token = Token(u"ewoe")
        assert self.indexer.byte_pair_encode(token) == [u'ew', u'oe</w>']

        # Prefer "ew" to "we"
        token = Token(u"ewe")
        assert self.indexer.byte_pair_encode(token) == [u'ew', u'e</w>']

        # Prefer ending a word
        token = Token(u"eee")
        assert self.indexer.byte_pair_encode(token) == [u'e', u'ee</w>']

        # Encodes up to a single symbol when appropriate
        token = Token(u"woe")
        assert self.indexer.byte_pair_encode(token) == [u'woe</w>']

    def test_tokens_to_indices(self):
        tokens = [Token(u'ewoe'), Token(u'woe'), Token(u'ewe'), Token(u'ee')]

        indices = self.indexer.tokens_to_indices(tokens, None, u'test')

        assert set(indices.keys()) == set([u"test", u"test-offsets", u"mask"])

        text_tokens = indices[u'test']
        offsets = indices[u'test-offsets']

        assert text_tokens[:6] == [
                self.indexer.encoder.get(symbol, 0)
                for symbol in [u'ew', u'oe</w>'] + [u'woe</w>'] + [u'ew', u'e</w>'] + [u'ee</w>']
        ]

        assert offsets == [
                1,  # end of first word
                2,  # end of second word
                4,  # end of third word
                5,  # end of last word
        ]

    def test_raises_with_too_long_sentence(self):
        tokens = [Token(u'a') for _ in range(513)]

        with pytest.raises(RuntimeError):
            self.indexer.tokens_to_indices(tokens, None, u'should-fail')
