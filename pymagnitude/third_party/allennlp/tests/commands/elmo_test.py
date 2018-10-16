# -*- coding: utf-8 -*-
# pylint: disable=no-self-use,invalid-name



from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import json
import os
import pathlib
import sys
import tempfile
import warnings
from io import open
try:
    from itertools import izip
except:
    izip = zip


with warnings.catch_warnings():
    warnings.filterwarnings(u"ignore", category=FutureWarning)
    import h5py
import numpy
import pytest

from allennlp.common.checks import ConfigurationError
from allennlp.commands import main
from allennlp.commands.elmo import ElmoEmbedder
from allennlp.tests.modules.elmo_test import ElmoTestCase


class TestElmoCommand(ElmoTestCase):
    def setUp(self):
        super(TestElmoCommand, self).setUp()
        self.tempdir = pathlib.Path(tempfile.mkdtemp())
        self.sentences_path = unicode(self.tempdir / u"sentences.txt")
        self.output_path = unicode(self.tempdir / u"output.txt")

    def test_all_embedding_works(self):
        sentence = u"Michael went to the store to buy some eggs ."
        with open(self.sentences_path, u'w') as f:
            f.write(sentence)

        sys.argv = [u"run.py",  # executable
                    u"elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    u"--all",
                    u"--options-file",
                    self.options_file,
                    u"--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        expected_embedding = embedder.embed_sentence(sentence.split())

        with h5py.File(self.output_path, u'r') as h5py_file:
            assert set(h5py_file.keys()) == set([u"0", u"sentence_to_index"])
            # The vectors in the test configuration are smaller (32 length)
            embedding = h5py_file.get(u"0")
            assert embedding.shape == (3, len(sentence.split()), 32)
            numpy.testing.assert_allclose(embedding, expected_embedding, rtol=1e-4)
            assert json.loads(h5py_file.get(u"sentence_to_index")[0]) == {sentence: u"0"}

    def test_top_embedding_works(self):
        sentence = u"Michael went to the store to buy some eggs ."
        with open(self.sentences_path, u'w') as f:
            f.write(sentence)

        sys.argv = [u"run.py",  # executable
                    u"elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    u"--top",
                    u"--options-file",
                    self.options_file,
                    u"--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        expected_embedding = embedder.embed_sentence(sentence.split())[2]

        with h5py.File(self.output_path, u'r') as h5py_file:
            assert set(h5py_file.keys()) == set([u"0", u"sentence_to_index"])
            # The vectors in the test configuration are smaller (32 length)
            embedding = h5py_file.get(u"0")
            assert embedding.shape == (len(sentence.split()), 32)
            numpy.testing.assert_allclose(embedding, expected_embedding, rtol=1e-4)
            assert json.loads(h5py_file.get(u"sentence_to_index")[0]) == {sentence: u"0"}

    def test_average_embedding_works(self):
        sentence = u"Michael went to the store to buy some eggs ."
        with open(self.sentences_path, u'w') as f:
            f.write(sentence)

        sys.argv = [u"run.py",  # executable
                    u"elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    u"--average",
                    u"--options-file",
                    self.options_file,
                    u"--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        expected_embedding = embedder.embed_sentence(sentence.split())
        expected_embedding = (expected_embedding[0] + expected_embedding[1] + expected_embedding[2]) / 3

        with h5py.File(self.output_path, u'r') as h5py_file:
            assert set(h5py_file.keys()) == set([u"0", u"sentence_to_index"])
            # The vectors in the test configuration are smaller (32 length)
            embedding = h5py_file.get(u"0")
            assert embedding.shape == (len(sentence.split()), 32)
            numpy.testing.assert_allclose(embedding, expected_embedding, rtol=1e-4)
            assert json.loads(h5py_file.get(u"sentence_to_index")[0]) == {sentence: u"0"}

    def test_batch_embedding_works(self):
        sentences = [
                u"Michael went to the store to buy some eggs .",
                u"Joel rolled down the street on his skateboard .",
                u"test / this is a first sentence",
                u"Take a look , then , at Tuesday 's elections in New York City , New Jersey and Virginia :"
        ]

        with open(self.sentences_path, u'w') as f:
            for line in sentences:
                f.write(line + u'\n')

        sys.argv = [u"run.py",  # executable
                    u"elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    u"--all",
                    u"--options-file",
                    self.options_file,
                    u"--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        with h5py.File(self.output_path, u'r') as h5py_file:
            assert set(h5py_file.keys()) == set([u"0", u"1", u"2", u"3", u"sentence_to_index"])
            # The vectors in the test configuration are smaller (32 length)
            for sentence_id, sentence in izip([u"0", u"1", u"2", u"3"], sentences):
                assert h5py_file.get(sentence_id).shape == (3, len(sentence.split()), 32)
            assert (json.loads(h5py_file.get(u"sentence_to_index")[0]) ==
                    dict((sentences[i], unicode(i)) for i in range(len(sentences))))

    def test_batch_embedding_works_with_sentences_as_keys(self):
        sentences = [
                u"Michael went to the store to buy some eggs .",
                u"Joel rolled down the street on his skateboard ."
        ]

        with open(self.sentences_path, u'w') as f:
            for line in sentences:
                f.write(line + u'\n')

        sys.argv = [u"run.py",  # executable
                    u"elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    u"--all",
                    u"--options-file",
                    self.options_file,
                    u"--weight-file",
                    self.weight_file,
                    u"--use-sentence-keys"]
        main()

        assert os.path.exists(self.output_path)

        with h5py.File(self.output_path, u'r') as h5py_file:
            assert set(h5py_file.keys()) == set(sentences)
            # The vectors in the test configuration are smaller (32 length)
            for sentence in sentences:
                assert h5py_file.get(sentence).shape == (3, len(sentence.split()), 32)

    def test_batch_embedding_works_with_forget_sentences(self):
        sentences = [
                u"Michael went to the store to buy some eggs .",
                u"Joel rolled down the street on his skateboard .",
                u"test / this is a first sentence",
                u"Take a look , then , at Tuesday 's elections in New York City , New Jersey and Virginia :"
        ]

        with open(self.sentences_path, u'w') as f:
            for line in sentences:
                f.write(line + u'\n')

        sys.argv = [u"run.py",  # executable
                    u"elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    u"--all",
                    u"--options-file",
                    self.options_file,
                    u"--weight-file",
                    self.weight_file,
                    u"--forget-sentences"]

        main()

        assert os.path.exists(self.output_path)

        with h5py.File(self.output_path, u'r') as h5py_file:
            assert set(h5py_file.keys()) == set([u"0", u"1", u"2", u"3"])
            # The vectors in the test configuration are smaller (32 length)
            for sentence_id, sentence in izip([u"0", u"1", u"2", u"3"], sentences):
                assert h5py_file.get(sentence_id).shape == (3, len(sentence.split()), 32)

    def test_duplicate_sentences(self):
        sentences = [
                u"Michael went to the store to buy some eggs .",
                u"Michael went to the store to buy some eggs .",
        ]

        with open(self.sentences_path, u'w') as f:
            for line in sentences:
                f.write(line + u'\n')

        sys.argv = [u"run.py",  # executable
                    u"elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    u"--all",
                    u"--options-file",
                    self.options_file,
                    u"--weight-file",
                    self.weight_file]

        main()

        assert os.path.exists(self.output_path)

        with h5py.File(self.output_path, u'r') as h5py_file:
            assert len(list(h5py_file.keys())) == 3
            assert set(h5py_file.keys()) == set([u"0", u"1", u"sentence_to_index"])
            # The vectors in the test configuration are smaller (32 length)
            for sentence_id, sentence in izip([u"0", u"1"], sentences):
                assert h5py_file.get(sentence_id).shape == (3, len(sentence.split()), 32)

    def test_empty_sentences_raise_errors(self):
        sentences = [
                u"A",
                u"",
                u"",
                u"B"
        ]

        with open(self.sentences_path, u'w') as f:
            for line in sentences:
                f.write(line + u'\n')

        sys.argv = [u"run.py",  # executable
                    u"elmo",  # command
                    self.sentences_path,
                    self.output_path,
                    u"--all",
                    u"--options-file",
                    self.options_file,
                    u"--weight-file",
                    self.weight_file]
        with pytest.raises(ConfigurationError):
            main()


class TestElmoEmbedder(ElmoTestCase):
    def test_embeddings_are_as_expected(self):
        loaded_sentences, loaded_embeddings = self._load_sentences_embeddings()

        assert len(loaded_sentences) == len(loaded_embeddings)
        batch_size = len(loaded_sentences)

        # The sentences and embeddings are organized in an idiosyncratic way TensorFlow handles batching.
        # We are going to reorganize them linearly so they can be grouped into batches by AllenNLP.
        sentences = []
        expected_embeddings = []
        for batch_number in range(len(loaded_sentences[0])):
            for index in range(batch_size):
                sentences.append(loaded_sentences[index][batch_number].split())
                expected_embeddings.append(loaded_embeddings[index][batch_number])

        assert len(expected_embeddings) == len(sentences)

        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        embeddings = list(embedder.embed_sentences(sentences, batch_size))

        assert len(embeddings) == len(sentences)

        for tensor, expected in izip(embeddings, expected_embeddings):
            numpy.testing.assert_array_almost_equal(tensor[2], expected)

    def test_embed_batch_is_empty_sentence(self):
        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        embeddings = embedder.embed_sentence([])

        assert embeddings.shape == (3, 0, 1024)

    def test_embed_batch_contains_empty_sentence(self):
        embedder = ElmoEmbedder(options_file=self.options_file, weight_file=self.weight_file)
        embeddings = list(embedder.embed_sentences([u"This is a test".split(), []]))

        assert len(embeddings) == 2
