


from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import codecs
import gzip
import zipfile
from copy import deepcopy
import copy
import shutil
import pytest
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance, Token
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import CharacterTokenizer
from allennlp.data.vocabulary import (Vocabulary, _NamespaceDependentDefaultDict,
                                      DEFAULT_OOV_TOKEN, _read_pretrained_tokens)
from allennlp.common.params import Params
from allennlp.common.checks import ConfigurationError
from allennlp.modules.token_embedders.embedding import format_embeddings_file_uri


class TestVocabulary(AllenNlpTestCase):
    # pylint: disable=no-self-use, invalid-name, too-many-public-methods, protected-access

    def setUp(self):
        token_indexer = SingleIdTokenIndexer(u"tokens")
        text_field = TextField([Token(t) for t in [u"a", u"a", u"a", u"a", u"b", u"b", u"c", u"c", u"c"]],
                               {u"tokens": token_indexer})
        self.instance = Instance({u"text": text_field})
        self.dataset = Batch([self.instance])
        super(TestVocabulary, self).setUp()

    def test_from_dataset_respects_max_vocab_size_single_int(self):
        max_vocab_size = 1
        vocab = Vocabulary.from_instances(self.dataset, max_vocab_size=max_vocab_size)
        words = list(vocab.get_index_to_token_vocabulary().values())
        # Additional 2 tokens are '@@PADDING@@' and '@@UNKNOWN@@' by default
        assert len(words) == max_vocab_size + 2

        vocab = Vocabulary.from_instances(self.dataset, min_count=None)
        words = list(vocab.get_index_to_token_vocabulary().values())
        assert len(words) == 5

    def test_from_dataset_respects_min_count(self):
        vocab = Vocabulary.from_instances(self.dataset, min_count={u'tokens': 4})
        words = list(vocab.get_index_to_token_vocabulary().values())
        assert u'a' in words
        assert u'b' not in words
        assert u'c' not in words

        vocab = Vocabulary.from_instances(self.dataset, min_count=None)
        words = list(vocab.get_index_to_token_vocabulary().values())
        assert u'a' in words
        assert u'b' in words
        assert u'c' in words

    def test_from_dataset_respects_exclusive_embedding_file(self):
        embeddings_filename = unicode(self.TEST_DIR / u"embeddings.gz")
        with gzip.open(embeddings_filename, u'wb') as embeddings_file:
            embeddings_file.write(u"a 1.0 2.3 -1.0\n".encode(u'utf-8'))
            embeddings_file.write(u"b 0.1 0.4 -4.0\n".encode(u'utf-8'))

        vocab = Vocabulary.from_instances(self.dataset,
                                          min_count={u'tokens': 4},
                                          pretrained_files={u'tokens': embeddings_filename},
                                          only_include_pretrained_words=True)
        words = list(vocab.get_index_to_token_vocabulary().values())
        assert u'a' in words
        assert u'b' not in words
        assert u'c' not in words

        vocab = Vocabulary.from_instances(self.dataset,
                                          pretrained_files={u'tokens': embeddings_filename},
                                          only_include_pretrained_words=True)
        words = list(vocab.get_index_to_token_vocabulary().values())
        assert u'a' in words
        assert u'b' in words
        assert u'c' not in words

    def test_from_dataset_respects_inclusive_embedding_file(self):
        embeddings_filename = unicode(self.TEST_DIR / u"embeddings.gz")
        with gzip.open(embeddings_filename, u'wb') as embeddings_file:
            embeddings_file.write(u"a 1.0 2.3 -1.0\n".encode(u'utf-8'))
            embeddings_file.write(u"b 0.1 0.4 -4.0\n".encode(u'utf-8'))

        vocab = Vocabulary.from_instances(self.dataset,
                                          min_count={u'tokens': 4},
                                          pretrained_files={u'tokens': embeddings_filename},
                                          only_include_pretrained_words=False)
        words = list(vocab.get_index_to_token_vocabulary().values())
        assert u'a' in words
        assert u'b' in words
        assert u'c' not in words

        vocab = Vocabulary.from_instances(self.dataset,
                                          pretrained_files={u'tokens': embeddings_filename},
                                          only_include_pretrained_words=False)
        words = list(vocab.get_index_to_token_vocabulary().values())
        assert u'a' in words
        assert u'b' in words
        assert u'c' in words

    def test_add_word_to_index_gives_consistent_results(self):
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace(u"word")
        assert u"word" in list(vocab.get_index_to_token_vocabulary().values())
        assert vocab.get_token_index(u"word") == word_index
        assert vocab.get_token_from_index(word_index) == u"word"
        assert vocab.get_vocab_size() == initial_vocab_size + 1

        # Now add it again, and make sure nothing changes.
        vocab.add_token_to_namespace(u"word")
        assert u"word" in list(vocab.get_index_to_token_vocabulary().values())
        assert vocab.get_token_index(u"word") == word_index
        assert vocab.get_token_from_index(word_index) == u"word"
        assert vocab.get_vocab_size() == initial_vocab_size + 1

    def test_namespaces(self):
        vocab = Vocabulary()
        initial_vocab_size = vocab.get_vocab_size()
        word_index = vocab.add_token_to_namespace(u"word", namespace=u'1')
        assert u"word" in list(vocab.get_index_to_token_vocabulary(namespace=u'1').values())
        assert vocab.get_token_index(u"word", namespace=u'1') == word_index
        assert vocab.get_token_from_index(word_index, namespace=u'1') == u"word"
        assert vocab.get_vocab_size(namespace=u'1') == initial_vocab_size + 1

        # Now add it again, in a different namespace and a different word, and make sure it's like
        # new.
        word2_index = vocab.add_token_to_namespace(u"word2", namespace=u'2')
        word_index = vocab.add_token_to_namespace(u"word", namespace=u'2')
        assert u"word" in list(vocab.get_index_to_token_vocabulary(namespace=u'2').values())
        assert u"word2" in list(vocab.get_index_to_token_vocabulary(namespace=u'2').values())
        assert vocab.get_token_index(u"word", namespace=u'2') == word_index
        assert vocab.get_token_index(u"word2", namespace=u'2') == word2_index
        assert vocab.get_token_from_index(word_index, namespace=u'2') == u"word"
        assert vocab.get_token_from_index(word2_index, namespace=u'2') == u"word2"
        assert vocab.get_vocab_size(namespace=u'2') == initial_vocab_size + 2

    def test_namespace_dependent_default_dict(self):
        default_dict = _NamespaceDependentDefaultDict([u"bar", u"*baz"], lambda: 7, lambda: 3)
        # 'foo' is not a padded namespace
        assert default_dict[u"foo"] == 7
        # "baz" is a direct match with a padded namespace
        assert default_dict[u"baz"] == 3
        # the following match the wildcard "*baz"
        assert default_dict[u"bar"] == 3
        assert default_dict[u"foobaz"] == 3

    def test_unknown_token(self):
        # pylint: disable=protected-access
        # We're putting this behavior in a test so that the behavior is documented.  There is
        # solver code that depends in a small way on how we treat the unknown token, so any
        # breaking change to this behavior should break a test, so you know you've done something
        # that needs more consideration.
        vocab = Vocabulary()
        oov_token = vocab._oov_token
        oov_index = vocab.get_token_index(oov_token)
        assert oov_index == 1
        assert vocab.get_token_index(u"unseen word") == oov_index

    def test_set_from_file_reads_padded_files(self):
        # pylint: disable=protected-access
        vocab_filename = self.TEST_DIR / u'vocab_file'
        with codecs.open(vocab_filename, u'w', u'utf-8') as vocab_file:
            vocab_file.write(u'<S>\n')
            vocab_file.write(u'</S>\n')
            vocab_file.write(u'<UNK>\n')
            vocab_file.write(u'a\n')
            vocab_file.write(u'tricky\x0bchar\n')
            vocab_file.write(u'word\n')
            vocab_file.write(u'another\n')

        vocab = Vocabulary()
        vocab.set_from_file(vocab_filename, is_padded=True, oov_token=u"<UNK>")

        assert vocab._oov_token == DEFAULT_OOV_TOKEN
        assert vocab.get_token_index(u"random string") == 3
        assert vocab.get_token_index(u"<S>") == 1
        assert vocab.get_token_index(u"</S>") == 2
        assert vocab.get_token_index(DEFAULT_OOV_TOKEN) == 3
        assert vocab.get_token_index(u"a") == 4
        assert vocab.get_token_index(u"tricky\x0bchar") == 5
        assert vocab.get_token_index(u"word") == 6
        assert vocab.get_token_index(u"another") == 7
        assert vocab.get_token_from_index(0) == vocab._padding_token
        assert vocab.get_token_from_index(1) == u"<S>"
        assert vocab.get_token_from_index(2) == u"</S>"
        assert vocab.get_token_from_index(3) == DEFAULT_OOV_TOKEN
        assert vocab.get_token_from_index(4) == u"a"
        assert vocab.get_token_from_index(5) == u"tricky\x0bchar"
        assert vocab.get_token_from_index(6) == u"word"
        assert vocab.get_token_from_index(7) == u"another"

    def test_set_from_file_reads_non_padded_files(self):
        # pylint: disable=protected-access
        vocab_filename = self.TEST_DIR / u'vocab_file'
        with codecs.open(vocab_filename, u'w', u'utf-8') as vocab_file:
            vocab_file.write(u'B-PERS\n')
            vocab_file.write(u'I-PERS\n')
            vocab_file.write(u'O\n')
            vocab_file.write(u'B-ORG\n')
            vocab_file.write(u'I-ORG\n')

        vocab = Vocabulary()
        vocab.set_from_file(vocab_filename, is_padded=False, namespace=u'tags')
        assert vocab.get_token_index(u"B-PERS", namespace=u'tags') == 0
        assert vocab.get_token_index(u"I-PERS", namespace=u'tags') == 1
        assert vocab.get_token_index(u"O", namespace=u'tags') == 2
        assert vocab.get_token_index(u"B-ORG", namespace=u'tags') == 3
        assert vocab.get_token_index(u"I-ORG", namespace=u'tags') == 4
        assert vocab.get_token_from_index(0, namespace=u'tags') == u"B-PERS"
        assert vocab.get_token_from_index(1, namespace=u'tags') == u"I-PERS"
        assert vocab.get_token_from_index(2, namespace=u'tags') == u"O"
        assert vocab.get_token_from_index(3, namespace=u'tags') == u"B-ORG"
        assert vocab.get_token_from_index(4, namespace=u'tags') == u"I-ORG"

    def test_saving_and_loading(self):
        # pylint: disable=protected-access
        vocab_dir = self.TEST_DIR / u'vocab_save'

        vocab = Vocabulary(non_padded_namespaces=[u"a", u"c"])
        vocab.add_token_to_namespace(u"a0", namespace=u"a")  # non-padded, should start at 0
        vocab.add_token_to_namespace(u"a1", namespace=u"a")
        vocab.add_token_to_namespace(u"a2", namespace=u"a")
        vocab.add_token_to_namespace(u"b2", namespace=u"b")  # padded, should start at 2
        vocab.add_token_to_namespace(u"b3", namespace=u"b")

        vocab.save_to_files(vocab_dir)
        vocab2 = Vocabulary.from_files(vocab_dir)

        assert vocab2._non_padded_namespaces == set([u"a", u"c"])

        # Check namespace a.
        assert vocab2.get_vocab_size(namespace=u'a') == 3
        assert vocab2.get_token_from_index(0, namespace=u'a') == u'a0'
        assert vocab2.get_token_from_index(1, namespace=u'a') == u'a1'
        assert vocab2.get_token_from_index(2, namespace=u'a') == u'a2'
        assert vocab2.get_token_index(u'a0', namespace=u'a') == 0
        assert vocab2.get_token_index(u'a1', namespace=u'a') == 1
        assert vocab2.get_token_index(u'a2', namespace=u'a') == 2

        # Check namespace b.
        assert vocab2.get_vocab_size(namespace=u'b') == 4  # (unk + padding + two tokens)
        assert vocab2.get_token_from_index(0, namespace=u'b') == vocab._padding_token
        assert vocab2.get_token_from_index(1, namespace=u'b') == vocab._oov_token
        assert vocab2.get_token_from_index(2, namespace=u'b') == u'b2'
        assert vocab2.get_token_from_index(3, namespace=u'b') == u'b3'
        assert vocab2.get_token_index(vocab._padding_token, namespace=u'b') == 0
        assert vocab2.get_token_index(vocab._oov_token, namespace=u'b') == 1
        assert vocab2.get_token_index(u'b2', namespace=u'b') == 2
        assert vocab2.get_token_index(u'b3', namespace=u'b') == 3

        # Check the dictionaries containing the reverse mapping are identical.
        assert vocab.get_index_to_token_vocabulary(u"a") == vocab2.get_index_to_token_vocabulary(u"a")
        assert vocab.get_index_to_token_vocabulary(u"b") == vocab2.get_index_to_token_vocabulary(u"b")

    def test_saving_and_loading_works_with_byte_encoding(self):
        # We're going to set a vocabulary from a TextField using byte encoding, index it, save the
        # vocab, load the vocab, then index the text field again, and make sure we get the same
        # result.
        tokenizer = CharacterTokenizer(byte_encoding=u'utf-8')
        token_indexer = TokenCharactersIndexer(character_tokenizer=tokenizer)
        tokens = [Token(t) for t in [u"Øyvind", u"für", u"汉字"]]
        text_field = TextField(tokens, {u"characters": token_indexer})
        dataset = Batch([Instance({u"sentence": text_field})])
        vocab = Vocabulary.from_instances(dataset)
        text_field.index(vocab)
        indexed_tokens = deepcopy(text_field._indexed_tokens)  # pylint: disable=protected-access

        vocab_dir = self.TEST_DIR / u'vocab_save'
        vocab.save_to_files(vocab_dir)
        vocab2 = Vocabulary.from_files(vocab_dir)
        text_field2 = TextField(tokens, {u"characters": token_indexer})
        text_field2.index(vocab2)
        indexed_tokens2 = deepcopy(text_field2._indexed_tokens)  # pylint: disable=protected-access
        assert indexed_tokens == indexed_tokens2

    def test_from_params(self):
        # Save a vocab to check we can load it from_params.
        vocab_dir = self.TEST_DIR / u'vocab_save'
        vocab = Vocabulary(non_padded_namespaces=[u"a", u"c"])
        vocab.add_token_to_namespace(u"a0", namespace=u"a")  # non-padded, should start at 0
        vocab.add_token_to_namespace(u"a1", namespace=u"a")
        vocab.add_token_to_namespace(u"a2", namespace=u"a")
        vocab.add_token_to_namespace(u"b2", namespace=u"b")  # padded, should start at 2
        vocab.add_token_to_namespace(u"b3", namespace=u"b")
        vocab.save_to_files(vocab_dir)

        params = Params({u"directory_path": vocab_dir})
        vocab2 = Vocabulary.from_params(params)
        assert vocab.get_index_to_token_vocabulary(u"a") == vocab2.get_index_to_token_vocabulary(u"a")
        assert vocab.get_index_to_token_vocabulary(u"b") == vocab2.get_index_to_token_vocabulary(u"b")

        # Test case where we build a vocab from a dataset.
        vocab2 = Vocabulary.from_params(Params({}), self.dataset)
        assert vocab2.get_index_to_token_vocabulary(u"tokens") == {0: u'@@PADDING@@',
                                                                  1: u'@@UNKNOWN@@',
                                                                  2: u'a', 3: u'c', 4: u'b'}
        # Test from_params raises when we have neither a dataset and a vocab_directory.
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(Params({}))

        # Test from_params raises when there are any other dict keys
        # present apart from 'directory_path' and we aren't calling from_dataset.
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(Params({u"directory_path": vocab_dir, u"min_count": {u'tokens': 2}}))

    def test_from_params_adds_tokens_to_vocab(self):
        vocab = Vocabulary.from_params(Params({u'tokens_to_add': {u'tokens': [u'q', u'x', u'z']}}), self.dataset)
        assert vocab.get_index_to_token_vocabulary(u"tokens") == {0: u'@@PADDING@@',
                                                                 1: u'@@UNKNOWN@@',
                                                                 2: u'a', 3: u'c', 4: u'b',
                                                                 5: u'q', 6: u'x', 7: u'z'}

    def test_valid_vocab_extension(self):
        vocab_dir = self.TEST_DIR / u'vocab_save'
        extension_ways = [u"from_params", u"extend_from_instances"]
        # Test: padded/non-padded common namespaces are extending appropriately
        non_padded_namespaces_list = [[], [u"tokens"]]
        for non_padded_namespaces in non_padded_namespaces_list:
            original_vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
            original_vocab.add_token_to_namespace(u"d", namespace=u"tokens")
            original_vocab.add_token_to_namespace(u"a", namespace=u"tokens")
            original_vocab.add_token_to_namespace(u"b", namespace=u"tokens")
            text_field = TextField([Token(t) for t in [u"a", u"d", u"c", u"e"]],
                                   {u"tokens": SingleIdTokenIndexer(u"tokens")})
            instances = Batch([Instance({u"text": text_field})])
            for way in extension_ways:
                if way == u"extend_from_instances":
                    extended_vocab = copy.copy(original_vocab)
                    params = Params({u"non_padded_namespaces": non_padded_namespaces})
                    extended_vocab.extend_from_instances(params, instances)
                else:
                    shutil.rmtree(vocab_dir, ignore_errors=True)
                    original_vocab.save_to_files(vocab_dir)
                    params = Params({u"directory_path": vocab_dir, u"extend": True,
                                     u"non_padded_namespaces": non_padded_namespaces})
                    extended_vocab = Vocabulary.from_params(params, instances)

                extra_count = 2 if extended_vocab.is_padded(u"tokens") else 0
                assert extended_vocab.get_token_index(u"d", u"tokens") == 0 + extra_count
                assert extended_vocab.get_token_index(u"a", u"tokens") == 1 + extra_count
                assert extended_vocab.get_token_index(u"b", u"tokens") == 2 + extra_count

                assert extended_vocab.get_token_index(u"c", u"tokens") # should be present
                assert extended_vocab.get_token_index(u"e", u"tokens") # should be present

                assert extended_vocab.get_vocab_size(u"tokens") == 5 + extra_count

        # Test: padded/non-padded non-common namespaces are extending appropriately
        non_padded_namespaces_list = [[],
                                      [u"tokens1"],
                                      [u"tokens1", u"tokens2"]]
        for non_padded_namespaces in non_padded_namespaces_list:
            original_vocab = Vocabulary(non_padded_namespaces=non_padded_namespaces)
            original_vocab.add_token_to_namespace(u"a", namespace=u"tokens1") # index2
            text_field = TextField([Token(t) for t in [u"b"]],
                                   {u"tokens2": SingleIdTokenIndexer(u"tokens2")})
            instances = Batch([Instance({u"text": text_field})])

            for way in extension_ways:
                if way == u"extend_from_instances":
                    extended_vocab = copy.copy(original_vocab)
                    params = Params({u"non_padded_namespaces": non_padded_namespaces})
                    extended_vocab.extend_from_instances(params, instances)
                else:
                    shutil.rmtree(vocab_dir, ignore_errors=True)
                    original_vocab.save_to_files(vocab_dir)
                    params = Params({u"directory_path": vocab_dir, u"extend": True,
                                     u"non_padded_namespaces": non_padded_namespaces})
                    extended_vocab = Vocabulary.from_params(params, instances)

                # Should have two namespaces
                assert len(extended_vocab._token_to_index) == 2

                extra_count = 2 if extended_vocab.is_padded(u"tokens1") else 0
                assert extended_vocab.get_vocab_size(u"tokens1") == 1 + extra_count

                extra_count = 2 if extended_vocab.is_padded(u"tokens2") else 0
                assert extended_vocab.get_vocab_size(u"tokens2") == 1 + extra_count

    def test_invalid_vocab_extension(self):
        vocab_dir = self.TEST_DIR / u'vocab_save'
        original_vocab = Vocabulary(non_padded_namespaces=[u"tokens1"])
        original_vocab.add_token_to_namespace(u"a", namespace=u"tokens1")
        original_vocab.add_token_to_namespace(u"b", namespace=u"tokens1")
        original_vocab.add_token_to_namespace(u"p", namespace=u"tokens2")
        original_vocab.save_to_files(vocab_dir)
        text_field1 = TextField([Token(t) for t in [u"a" u"c"]],
                                {u"tokens1": SingleIdTokenIndexer(u"tokens1")})
        text_field2 = TextField([Token(t) for t in [u"p", u"q", u"r"]],
                                {u"tokens2": SingleIdTokenIndexer(u"tokens2")})
        instances = Batch([Instance({u"text1": text_field1, u"text2": text_field2})])

        # Following 2 should give error: token1 is non-padded in original_vocab but not in instances
        params = Params({u"directory_path": vocab_dir, u"extend": True,
                         u"non_padded_namespaces": []})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params, instances)
        with pytest.raises(ConfigurationError):
            extended_vocab = copy.copy(original_vocab)
            params = Params({u"non_padded_namespaces": []})
            extended_vocab.extend_from_instances(params, instances)
        with pytest.raises(ConfigurationError):
            extended_vocab = copy.copy(original_vocab)
            extended_vocab._extend(non_padded_namespaces=[],
                                   tokens_to_add={u"tokens1": [u"a"], u"tokens2": [u"p"]})

        # Following 2 should not give error: overlapping namespaces have same padding setting
        params = Params({u"directory_path": vocab_dir, u"extend": True,
                         u"non_padded_namespaces": [u"tokens1"]})
        Vocabulary.from_params(params, instances)
        extended_vocab = copy.copy(original_vocab)
        params = Params({u"non_padded_namespaces": [u"tokens1"]})
        extended_vocab.extend_from_instances(params, instances)
        extended_vocab = copy.copy(original_vocab)
        extended_vocab._extend(non_padded_namespaces=[u"tokens1"],
                               tokens_to_add={u"tokens1": [u"a"], u"tokens2": [u"p"]})

        # Following 2 should give error: token1 is padded in instances but not in original_vocab
        params = Params({u"directory_path": vocab_dir, u"extend": True,
                         u"non_padded_namespaces": [u"tokens1", u"tokens2"]})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params, instances)
        with pytest.raises(ConfigurationError):
            extended_vocab = copy.copy(original_vocab)
            params = Params({u"non_padded_namespaces": [u"tokens1", u"tokens2"]})
            extended_vocab.extend_from_instances(params, instances)
        with pytest.raises(ConfigurationError):
            extended_vocab = copy.copy(original_vocab)
            extended_vocab._extend(non_padded_namespaces=[u"tokens1", u"tokens2"],
                                   tokens_to_add={u"tokens1": [u"a"], u"tokens2": [u"p"]})

    def test_from_params_extend_config(self):

        vocab_dir = self.TEST_DIR / u'vocab_save'
        original_vocab = Vocabulary(non_padded_namespaces=[u"tokens"])
        original_vocab.add_token_to_namespace(u"a", namespace=u"tokens")
        original_vocab.save_to_files(vocab_dir)

        text_field = TextField([Token(t) for t in [u"a", u"b"]],
                               {u"tokens": SingleIdTokenIndexer(u"tokens")})
        instances = Batch([Instance({u"text": text_field})])

        # If you ask to extend vocab from `directory_path`, instances must be passed
        # in Vocabulary constructor, or else there is nothing to extend to.
        params = Params({u"directory_path": vocab_dir, u"extend": True})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params)

        # If you ask to extend vocab, `directory_path` key must be present in params,
        # or else there is nothing to extend from.
        params = Params({u"extend": True})
        with pytest.raises(ConfigurationError):
            _ = Vocabulary.from_params(params, instances)

    def test_from_params_valid_vocab_extension_thoroughly(self):
        u'''
        Tests for Valid Vocab Extension thoroughly: Vocab extension is valid
        when overlapping namespaces have same padding behaviour (padded/non-padded)
        Summary of namespace paddings in this test:
        original_vocab namespaces
            tokens0     padded
            tokens1     non-padded
            tokens2     padded
            tokens3     non-padded
        instances namespaces
            tokens0     padded
            tokens1     non-padded
            tokens4     padded
            tokens5     non-padded
        TypicalExtention example: (of tokens1 namespace)
        -> original_vocab index2token
           apple          #0->apple
           bat            #1->bat
           cat            #2->cat
        -> Token to be extended with: cat, an, apple, banana, atom, bat
        -> extended_vocab: index2token
           apple           #0->apple
           bat             #1->bat
           cat             #2->cat
           an              #3->an
           atom            #4->atom
           banana          #5->banana
        '''

        vocab_dir = self.TEST_DIR / u'vocab_save'
        original_vocab = Vocabulary(non_padded_namespaces=[u"tokens1", u"tokens3"])
        original_vocab.add_token_to_namespace(u"apple", namespace=u"tokens0") # index:2
        original_vocab.add_token_to_namespace(u"bat", namespace=u"tokens0")   # index:3
        original_vocab.add_token_to_namespace(u"cat", namespace=u"tokens0")   # index:4

        original_vocab.add_token_to_namespace(u"apple", namespace=u"tokens1") # index:0
        original_vocab.add_token_to_namespace(u"bat", namespace=u"tokens1")   # index:1
        original_vocab.add_token_to_namespace(u"cat", namespace=u"tokens1")   # index:2

        original_vocab.add_token_to_namespace(u"a", namespace=u"tokens2") # index:0
        original_vocab.add_token_to_namespace(u"b", namespace=u"tokens2") # index:1
        original_vocab.add_token_to_namespace(u"c", namespace=u"tokens2") # index:2

        original_vocab.add_token_to_namespace(u"p", namespace=u"tokens3") # index:0
        original_vocab.add_token_to_namespace(u"q", namespace=u"tokens3") # index:1

        original_vocab.save_to_files(vocab_dir)

        text_field0 = TextField([Token(t) for t in [u"cat", u"an", u"apple", u"banana", u"atom", u"bat"]],
                                {u"tokens0": SingleIdTokenIndexer(u"tokens0")})
        text_field1 = TextField([Token(t) for t in [u"cat", u"an", u"apple", u"banana", u"atom", u"bat"]],
                                {u"tokens1": SingleIdTokenIndexer(u"tokens1")})
        text_field4 = TextField([Token(t) for t in [u"l", u"m", u"n", u"o"]],
                                {u"tokens4": SingleIdTokenIndexer(u"tokens4")})
        text_field5 = TextField([Token(t) for t in [u"x", u"y", u"z"]],
                                {u"tokens5": SingleIdTokenIndexer(u"tokens5")})
        instances = Batch([Instance({u"text0": text_field0, u"text1": text_field1,
                                     u"text4": text_field4, u"text5": text_field5})])

        params = Params({u"directory_path": vocab_dir,
                         u"extend": True,
                         u"non_padded_namespaces": [u"tokens1", u"tokens5"]})
        extended_vocab = Vocabulary.from_params(params, instances)

        # namespaces: tokens0, tokens1 is common.
        # tokens2, tokens3 only vocab has. tokens4, tokens5 only instances
        extended_namespaces = set([*extended_vocab._token_to_index])
        assert extended_namespaces == set(u"tokens{}".format(i) for i in range(6))

        # # Check that _non_padded_namespaces list is consistent after extension
        assert extended_vocab._non_padded_namespaces == set([u"tokens1", u"tokens3", u"tokens5"])

        # # original_vocab["tokens1"] has 3 tokens, instances of "tokens1" ns has 5 tokens. 2 overlapping
        assert extended_vocab.get_vocab_size(u"tokens1") == 6
        assert extended_vocab.get_vocab_size(u"tokens0") == 8 # 2 extra overlapping because padded

        # namespace tokens3, tokens4 was only in original_vocab,
        # and its token count should be same in extended_vocab
        assert extended_vocab.get_vocab_size(u"tokens2") == original_vocab.get_vocab_size(u"tokens2")
        assert extended_vocab.get_vocab_size(u"tokens3") == original_vocab.get_vocab_size(u"tokens3")

        # namespace tokens2 was only in instances,
        # and its token count should be same in extended_vocab
        assert extended_vocab.get_vocab_size(u"tokens4") == 6 # l,m,n,o + oov + padding
        assert extended_vocab.get_vocab_size(u"tokens5") == 3 # x,y,z

        # Word2index mapping of all words in all namespaces of original_vocab
        # should be maintained in extended_vocab
        for namespace, token2index in list(original_vocab._token_to_index.items()):
            for token, _ in list(token2index.items()):
                vocab_index = original_vocab.get_token_index(token, namespace)
                extended_vocab_index = extended_vocab.get_token_index(token, namespace)
                assert vocab_index == extended_vocab_index
        # And same for Index2Word mapping
        for namespace, index2token in list(original_vocab._index_to_token.items()):
            for index, _ in list(index2token.items()):
                vocab_token = original_vocab.get_token_from_index(index, namespace)
                extended_vocab_token = extended_vocab.get_token_from_index(index, namespace)
                assert vocab_token == extended_vocab_token

        # Manual Print Check
        # original_vocab._token_to_index :>
        # {
        #   "tokens0": {"@@PADDING@@":0,"@@UNKNOWN@@":1,"apple":2,"bat":3,"cat":4},
        #   "tokens1": {"apple": 0,"bat":1,"cat":2},
        #   "tokens2": {"@@PADDING@@":0,"@@UNKNOWN@@":1,"a":2,"b":3,"c": 4},
        #   "tokens3": {"p":0,"q":1}
        # }
        # extended_vocab._token_to_index :>
        # {
        #   "tokens0": {"@@PADDING@@": 0,"@@UNKNOWN@@": 1,
        #               "apple": 2,"bat": 3,"cat": 4,"an": 5,"banana": 6,"atom": 7},
        #   "tokens1": {"apple": 0,"bat": 1,"cat": 2,"an": 3,"banana": 4,"atom": 5},
        #   "tokens2": {"@@PADDING@@": 0,"@@UNKNOWN@@": 1,"a": 2,"b": 3,"c": 4},
        #   "tokens3": {"p": 0,"q": 1},
        #   "tokens4": {"@@PADDING@@": 0,"@@UNKNOWN@@": 1,"l": 2,"m": 3,"n": 4,"o": 5},
        #   "tokens5": {"x": 0,"y": 1,"z": 2}
        # }

    def test_vocab_can_print(self):
        vocab = Vocabulary(non_padded_namespaces=[u"a", u"c"])
        vocab.add_token_to_namespace(u"a0", namespace=u"a")
        vocab.add_token_to_namespace(u"a1", namespace=u"a")
        vocab.add_token_to_namespace(u"a2", namespace=u"a")
        vocab.add_token_to_namespace(u"b2", namespace=u"b")
        vocab.add_token_to_namespace(u"b3", namespace=u"b")
        print(vocab)

    def test_read_pretrained_words(self):
        # The fixture "fake_embeddings.5d.txt" was generated using the words in this random quote
        words = set(u"If you think you are too small to make a difference "
                    u"try to sleeping with a mosquito àèìòù".split(u' '))

        # Reading from a single (compressed) file or a single-file archive
        base_path = unicode(self.FIXTURES_ROOT / u"embeddings/fake_embeddings.5d.txt")
        for ext in [u'', u'.gz', u'.lzma', u'.bz2', u'.zip', u'.tar.gz']:
            file_path = base_path + ext
            words_read = _read_pretrained_tokens(file_path)
            assert words_read == words, "Wrong words for file {file_path}\n"\
                                        "   Read: {sorted(words_read)}\n"\
                                        "Correct: {sorted(words)}"

        # Reading from a multi-file archive
        base_path = unicode(self.FIXTURES_ROOT / u"embeddings/multi-file-archive")
        file_path = u'folder/fake_embeddings.5d.txt'
        for ext in [u'.zip', u'.tar.gz']:
            archive_path = base_path + ext
            embeddings_file_uri = format_embeddings_file_uri(archive_path, file_path)
            words_read = _read_pretrained_tokens(embeddings_file_uri)
            assert words_read == words, "Wrong words for file {archive_path}\n"\
                                        "   Read: {sorted(words_read)}\n"\
                                        "Correct: {sorted(words)}"

    def test_from_instances_exclusive_embeddings_file_inside_archive(self):
        u""" Just for ensuring there are no problems when reading pretrained tokens from an archive """
        # Read embeddings file from archive
        archive_path = unicode(self.TEST_DIR / u"embeddings-archive.zip")

        with zipfile.ZipFile(archive_path, u'w') as archive:
            file_path = u'embedding.3d.vec'
            with archive.open(file_path, u'w') as embeddings_file:
                embeddings_file.write(u"a 1.0 2.3 -1.0\n".encode(u'utf-8'))
                embeddings_file.write(u"b 0.1 0.4 -4.0\n".encode(u'utf-8'))

            with archive.open(u'dummy.vec', u'w') as dummy_file:
                dummy_file.write(u"c 1.0 2.3 -1.0 3.0\n".encode(u'utf-8'))

        embeddings_file_uri = format_embeddings_file_uri(archive_path, file_path)
        vocab = Vocabulary.from_instances(self.dataset,
                                          min_count={u'tokens': 4},
                                          pretrained_files={u'tokens': embeddings_file_uri},
                                          only_include_pretrained_words=True)

        words = set(vocab.get_index_to_token_vocabulary().values())
        assert u'a' in words
        assert u'b' not in words
        assert u'c' not in words

        vocab = Vocabulary.from_instances(self.dataset,
                                          pretrained_files={u'tokens': embeddings_file_uri},
                                          only_include_pretrained_words=True)
        words = set(vocab.get_index_to_token_vocabulary().values())
        assert u'a' in words
        assert u'b' in words
        assert u'c' not in words

    def test_registrability(self):

        class MyVocabulary(object):
            @classmethod
            def from_params(cls, params, instances=None):
                # pylint: disable=unused-argument
                return MyVocabulary()


                MyVocabulary = Vocabulary.register(u'my-vocabulary')(MyVocabulary)

        params = Params({u'type': u'my-vocabulary'})

        instance = Instance(fields={})

        vocab = Vocabulary.from_params(params=params, instances=[instance])

        assert isinstance(vocab, MyVocabulary)

    def test_max_vocab_size_dict(self):
        params = Params({
                u"max_vocab_size": {
                        u"tokens": 1,
                        u"characters": 20
                }
        })

        vocab = Vocabulary.from_params(params=params, instances=self.dataset)
        words = list(vocab.get_index_to_token_vocabulary().values())
        # Additional 2 tokens are '@@PADDING@@' and '@@UNKNOWN@@' by default
        assert len(words) == 3
