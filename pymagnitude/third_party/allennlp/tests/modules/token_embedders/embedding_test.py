# pylint: disable=no-self-use,invalid-name



from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import gzip
import warnings

import numpy
import pytest
import torch
from io import open
try:
    from itertools import izip
except:
    izip = zip

with warnings.catch_warnings():
    warnings.filterwarnings(u"ignore", category=FutureWarning)
    import h5py

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary
from allennlp.modules.token_embedders.embedding import (Embedding,
                                                        _read_pretrained_embeddings_file,
                                                        EmbeddingsTextFile,
                                                        format_embeddings_file_uri,
                                                        parse_embeddings_file_uri)


class TestEmbedding(AllenNlpTestCase):
    # pylint: disable=protected-access
    def test_get_embedding_layer_uses_correct_embedding_dim(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u'word1')
        vocab.add_token_to_namespace(u'word2')
        embeddings_filename = unicode(self.TEST_DIR / u"embeddings.gz")
        with gzip.open(embeddings_filename, u'wb') as embeddings_file:
            embeddings_file.write(u"word1 1.0 2.3 -1.0\n".encode(u'utf-8'))
            embeddings_file.write(u"word2 0.1 0.4 -4.0\n".encode(u'utf-8'))
        embedding_weights = _read_pretrained_embeddings_file(embeddings_filename, 3, vocab)
        assert tuple(embedding_weights.size()) == (4, 3)  # 4 because of padding and OOV
        with pytest.raises(ConfigurationError):
            _read_pretrained_embeddings_file(embeddings_filename, 4, vocab)

    def test_forward_works_with_projection_layer(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u'the')
        vocab.add_token_to_namespace(u'a')
        params = Params({
                u'pretrained_file': unicode(self.FIXTURES_ROOT / u'embeddings/glove.6B.300d.sample.txt.gz'),
                u'embedding_dim': 300,
                u'projection_dim': 20
                })
        embedding_layer = Embedding.from_params(vocab, params)
        input_tensor = torch.LongTensor([[3, 2, 1, 0]])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 4, 20)

        input_tensor = torch.LongTensor([[[3, 2, 1, 0]]])
        embedded = embedding_layer(input_tensor).data.numpy()
        assert embedded.shape == (1, 1, 4, 20)

    def test_embedding_layer_actually_initializes_word_vectors_correctly(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u"word")
        vocab.add_token_to_namespace(u"word2")
        unicode_space = u"\xa0\xa0\xa0\xa0\xa0\xa0\xa0\xa0"
        vocab.add_token_to_namespace(unicode_space)
        embeddings_filename = unicode(self.TEST_DIR / u"embeddings.gz")
        with gzip.open(embeddings_filename, u'wb') as embeddings_file:
            embeddings_file.write(u"word 1.0 2.3 -1.0\n".encode(u'utf-8'))
            embeddings_file.write("{unicode_space} 3.4 3.3 5.0\n".encode(u'utf-8'))
        params = Params({
                u'pretrained_file': embeddings_filename,
                u'embedding_dim': 3,
                })
        embedding_layer = Embedding.from_params(vocab, params)
        word_vector = embedding_layer.weight.data[vocab.get_token_index(u"word")]
        assert numpy.allclose(word_vector.numpy(), numpy.array([1.0, 2.3, -1.0]))
        word_vector = embedding_layer.weight.data[vocab.get_token_index(unicode_space)]
        assert numpy.allclose(word_vector.numpy(), numpy.array([3.4, 3.3, 5.0]))
        word_vector = embedding_layer.weight.data[vocab.get_token_index(u"word2")]
        assert not numpy.allclose(word_vector.numpy(), numpy.array([1.0, 2.3, -1.0]))

    def test_get_embedding_layer_initializes_unseen_words_randomly_not_zero(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u"word")
        vocab.add_token_to_namespace(u"word2")
        embeddings_filename = unicode(self.TEST_DIR / u"embeddings.gz")
        with gzip.open(embeddings_filename, u'wb') as embeddings_file:
            embeddings_file.write(u"word 1.0 2.3 -1.0\n".encode(u'utf-8'))
        params = Params({
                u'pretrained_file': embeddings_filename,
                u'embedding_dim': 3,
                })
        embedding_layer = Embedding.from_params(vocab, params)
        word_vector = embedding_layer.weight.data[vocab.get_token_index(u"word2")]
        assert not numpy.allclose(word_vector.numpy(), numpy.array([0.0, 0.0, 0.0]))

    def test_read_hdf5_format_file(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u"word")
        vocab.add_token_to_namespace(u"word2")
        embeddings_filename = unicode(self.TEST_DIR / u"embeddings.hdf5")
        embeddings = numpy.random.rand(vocab.get_vocab_size(), 5)
        with h5py.File(embeddings_filename, u'w') as fout:
            _ = fout.create_dataset(
                    u'embedding', embeddings.shape, dtype=u'float32', data=embeddings
            )

        params = Params({
                u'pretrained_file': embeddings_filename,
                u'embedding_dim': 5,
                })
        embedding_layer = Embedding.from_params(vocab, params)
        assert numpy.allclose(embedding_layer.weight.data.numpy(), embeddings)

    def test_read_hdf5_raises_on_invalid_shape(self):
        vocab = Vocabulary()
        vocab.add_token_to_namespace(u"word")
        embeddings_filename = unicode(self.TEST_DIR / u"embeddings.hdf5")
        embeddings = numpy.random.rand(vocab.get_vocab_size(), 10)
        with h5py.File(embeddings_filename, u'w') as fout:
            _ = fout.create_dataset(
                    u'embedding', embeddings.shape, dtype=u'float32', data=embeddings
            )

        params = Params({
                u'pretrained_file': embeddings_filename,
                u'embedding_dim': 5,
                })
        with pytest.raises(ConfigurationError):
            _ = Embedding.from_params(vocab, params)

    def test_read_embedding_file_inside_archive(self):
        token2vec = {
                u"think": torch.Tensor([0.143, 0.189, 0.555, 0.361, 0.472]),
                u"make": torch.Tensor([0.878, 0.651, 0.044, 0.264, 0.872]),
                u"difference": torch.Tensor([0.053, 0.162, 0.671, 0.110, 0.259]),
                u"àèìòù": torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])
                }
        vocab = Vocabulary()
        for token in token2vec:
            vocab.add_token_to_namespace(token)

        params = Params({
                u'pretrained_file': unicode(self.FIXTURES_ROOT / u'embeddings/multi-file-archive.zip'),
                u'embedding_dim': 5
                })
        with pytest.raises(ValueError, message=u"No ValueError when pretrained_file is a multi-file archive"):
            Embedding.from_params(vocab, params)

        for ext in [u'.zip', u'.tar.gz']:
            archive_path = unicode(self.FIXTURES_ROOT / u'embeddings/multi-file-archive') + ext
            file_uri = format_embeddings_file_uri(archive_path, u'folder/fake_embeddings.5d.txt')
            params = Params({
                    u'pretrained_file': file_uri,
                    u'embedding_dim': 5
                    })
            embeddings = Embedding.from_params(vocab, params).weight.data
            for tok, vec in list(token2vec.items()):
                i = vocab.get_token_index(tok)
                assert torch.equal(embeddings[i], vec), u'Problem with format ' + archive_path

    def test_embeddings_text_file(self):
        txt_path = unicode(self.FIXTURES_ROOT / u'utf-8_sample/utf-8_sample.txt')

        # This is for sure a correct way to read an utf-8 encoded text file
        with open(txt_path, u'rt', encoding=u'utf-8') as f:
            correct_text = f.read()

        # Check if we get the correct text on plain and compressed versions of the file
        paths = [txt_path] + [txt_path + ext for ext in [u'.gz', u'.zip']]
        for path in paths:
            with EmbeddingsTextFile(path) as f:
                text = f.read()
            assert text == correct_text, u"Test failed for file: " + path

        # Check for a file contained inside an archive with multiple files
        for ext in [u'.zip', u'.tar.gz', u'.tar.bz2', u'.tar.lzma']:
            archive_path = unicode(self.FIXTURES_ROOT / u'utf-8_sample/archives/utf-8') + ext
            file_uri = format_embeddings_file_uri(archive_path, u'folder/utf-8_sample.txt')
            with EmbeddingsTextFile(file_uri) as f:
                text = f.read()
            assert text == correct_text, u"Test failed for file: " + archive_path

        # Passing a second level path when not reading an archive
        with pytest.raises(ValueError):
            with EmbeddingsTextFile(format_embeddings_file_uri(txt_path, u'a/fake/path')):
                pass

    def test_embeddings_text_file_num_tokens(self):
        test_filename = unicode(self.TEST_DIR / u'temp_embeddings.vec')

        def check_num_tokens(first_line, expected_num_tokens):
            with open(test_filename, u'w') as f:
                f.write(first_line)
            with EmbeddingsTextFile(test_filename) as f:
                assert f.num_tokens == expected_num_tokens, "Wrong num tokens for line: {first_line}"

        valid_header_lines = [u'1000000 300', u'300 1000000', u'1000000']
        for line in valid_header_lines:
            check_num_tokens(line, expected_num_tokens=1_000_000)

        not_header_lines = [u'hello 1', u'hello 1 2', u'111 222 333', u'111 222 hello']
        for line in not_header_lines:
            check_num_tokens(line, expected_num_tokens=None)

    def test_decode_embeddings_file_uri(self):
        first_level_paths = [
                u'path/to/embeddings.gz',
                u'unicode/path/òàè+ù.vec',
                u'http://www.embeddings.com/path/to/embeddings.gz',
                u'http://www.embeddings.com/àèìòù?query=blabla.zip',
                ]
        second_level_paths = [
                u'path/to/glove.27B.300d.vec',
                u'òàè+ù.vec',
                u'crawl-300d-2M.vec'
                ]

        for simple_path in first_level_paths:
            assert parse_embeddings_file_uri(simple_path) == (simple_path, None)

        for path1, path2 in izip(first_level_paths, second_level_paths):
            uri = format_embeddings_file_uri(path1, path2)
            decoded = parse_embeddings_file_uri(uri)
            assert decoded == (path1, path2)
