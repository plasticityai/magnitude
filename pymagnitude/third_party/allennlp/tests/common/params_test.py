# pylint: disable=no-self-use,invalid-name,bad-continuation



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import json
import os
import re
import tempfile
from collections import OrderedDict

import pytest

from allennlp.common.params import Params, unflatten, with_fallback, parse_overrides
from allennlp.common.testing import AllenNlpTestCase
from io import open


class TestParams(AllenNlpTestCase):

    def test_load_from_file(self):
        filename = self.FIXTURES_ROOT / u'bidaf' / u'experiment.json'
        params = Params.from_file(filename)

        assert u"dataset_reader" in params
        assert u"trainer" in params

        model_params = params.pop(u"model")
        assert model_params.pop(u"type") == u"bidaf"

    def test_overrides(self):
        filename = self.FIXTURES_ROOT / u'bidaf' / u'experiment.json'
        overrides = u'{ "train_data_path": "FOO", "model": { "type": "BAR" },'\
                    u'"model.text_field_embedder.tokens.type": "BAZ" }'
        params = Params.from_file(filename, overrides)

        assert u"dataset_reader" in params
        assert u"trainer" in params
        assert params[u"train_data_path"] == u"FOO"

        model_params = params.pop(u"model")
        assert model_params.pop(u"type") == u"BAR"
        assert model_params[u"text_field_embedder"][u"tokens"][u"type"] == u"BAZ"

    def test_unflatten(self):
        flattened = {u"a.b.c": 1, u"a.b.d": 0, u"a.e.f.g.h": 2, u"b": 3}
        unflattened = unflatten(flattened)
        assert unflattened == {
            u"a": {
                u"b": {
                    u"c": 1,
                    u"d": 0
                },
                u"e": {
                    u"f": {
                        u"g": {
                            u"h": 2
                        }
                    }
                }
            },
            u"b": 3
        }

        # should do nothing to a non-flat dictionary
        assert unflatten(unflattened) == unflattened

    def test_with_fallback(self):
        preferred = {u"a": 1}
        fallback = {u"a": 0, u"b": 2}

        merged = with_fallback(preferred=preferred, fallback=fallback)
        assert merged == {u"a": 1, u"b": 2}

        # incompatibility is ok
        preferred = {u"a": {u"c": 3}}
        fallback = {u"a": 0, u"b": 2}
        merged = with_fallback(preferred=preferred, fallback=fallback)
        assert merged == {u"a": {u"c": 3}, u"b": 2}

        # goes deep
        preferred = {u"deep": {u"a": 1}}
        fallback = {u"deep": {u"a": 0, u"b": 2}}

        merged = with_fallback(preferred=preferred, fallback=fallback)
        assert merged == {u"deep": {u"a": 1, u"b": 2}}

    def test_parse_overrides(self):
        assert parse_overrides(u"") == {}
        assert parse_overrides(u"{}") == {}

        override_dict = parse_overrides(u'{"train_data": "/train", "trainer.num_epochs": 10}')
        assert override_dict == {
            u"train_data": u"/train",
            u"trainer": {
                u"num_epochs": 10
            }
        }

        params = with_fallback(
            preferred=override_dict,
            fallback={
                u"train_data": u"/test",
                u"model": u"bidaf",
                u"trainer": {u"num_epochs": 100, u"optimizer": u"sgd"}
            })

        assert params == {
            u"train_data": u"/train",
            u"model": u"bidaf",
            u"trainer": {u"num_epochs": 10, u"optimizer": u"sgd"}
        }

    def test_as_flat_dict(self):
        params = Params({
                u'a': 10,
                u'b': {
                        u'c': 20,
                        u'd': u'stuff'
                }
        }).as_flat_dict()

        assert params == {u'a': 10, u'b.c': 20, u'b.d': u'stuff'}

    def test_jsonnet_features(self):
        config_file = self.TEST_DIR / u'config.jsonnet'
        with open(config_file, u'w') as f:
            f.write(u"""{
                            // This example is copied straight from the jsonnet docs
                            person1: {
                                name: "Alice",
                                welcome: "Hello " + self.name + "!",
                            },
                            person2: self.person1 { name: "Bob" },
                        }""")

        params = Params.from_file(config_file)

        alice = params.pop(u"person1")
        bob = params.pop(u"person2")

        assert alice.as_dict() == {u"name": u"Alice", u"welcome": u"Hello Alice!"}
        assert bob.as_dict() == {u"name": u"Bob", u"welcome": u"Hello Bob!"}

        params.assert_empty(u"TestParams")


    def test_regexes_with_backslashes(self):
        bad_regex = self.TEST_DIR / u'bad_regex.jsonnet'
        good_regex = self.TEST_DIR / u'good_regex.jsonnet'

        with open(bad_regex, u'w') as f:
            f.write(ur'{"myRegex": "a\.b"}')

        with open(good_regex, u'w') as f:
            f.write(ur'{"myRegex": "a\\.b"}')

        with pytest.raises(RuntimeError):
            Params.from_file(bad_regex)

        params = Params.from_file(good_regex)
        regex = params[u'myRegex']

        assert re.match(regex, u"a.b")
        assert not re.match(regex, u"a-b")

        # Check roundtripping
        good_regex2 = self.TEST_DIR / u'good_regex2.jsonnet'
        with open(good_regex2, u'w') as f:
            f.write(json.dumps(params.as_dict()))
        params2 = Params.from_file(good_regex2)

        assert params.as_dict() == params2.as_dict()

    def test_env_var_substitution(self):
        substitutor = self.TEST_DIR / u'substitutor.jsonnet'
        key = u'TEST_ENV_VAR_SUBSTITUTION'

        assert os.environ.get(key) is None

        with open(substitutor, u'w') as f:
            f.write('{{"path": std.extVar("{key}")}}')

        # raises without environment variable set
        with pytest.raises(RuntimeError):
            Params.from_file(substitutor)

        os.environ[key] = u"PERFECT"

        params = Params.from_file(substitutor)
        assert params[u'path'] == u"PERFECT"

        del os.environ[key]

    @pytest.mark.xfail(not os.path.exists(AllenNlpTestCase.PROJECT_ROOT / u"training_config"),
                       reason=u"Training configs not installed with pip")
    def test_known_configs(self):
        configs = os.listdir(self.PROJECT_ROOT / u"training_config")

        # Our configs use environment variable substitution, and the _jsonnet parser
        # will fail if we don't pass it correct environment variables.
        forced_variables = [
            # constituency parser
            u'PTB_TRAIN_PATH', u'PTB_DEV_PATH', u'PTB_TEST_PATH',

            # srl_elmo_5.5B
            u'SRL_TRAIN_DATA_PATH', u'SRL_VALIDATION_DATA_PATH',

            # coref
            u'COREF_TRAIN_DATA_PATH', u'COREF_DEV_DATA_PATH', u'COREF_TEST_DATA_PATH',

            # ner
            u'NER_TRAIN_DATA_PATH', u'NER_TEST_A_PATH', u'NER_TEST_B_PATH'
        ]

        for var in forced_variables:
            os.environ[var] = os.environ.get(var) or unicode(self.TEST_DIR)

        for config in configs:
            try:
                Params.from_file(self.PROJECT_ROOT / u"training_config" / config)
            except Exception as e:
                raise AssertionError("unable to load params for {config}, because {e}")

        for var in forced_variables:
            if os.environ[var] == unicode(self.TEST_DIR):
                del os.environ[var]

    def test_add_file_to_archive(self):
        # Creates actual files since add_file_to_archive will throw an exception
        # if the file does not exist.
        tempdir = tempfile.mkdtemp()
        my_file = os.path.join(tempdir, u"my_file.txt")
        my_other_file = os.path.join(tempdir, u"my_other_file.txt")
        open(my_file, u'w').close()
        open(my_other_file, u'w').close()

        # Some nested classes just to exercise the ``from_params``
        # and ``add_file_to_archive`` methods.

        class C(object):
            def __init__(self, c_file     )        :
                self.c_file = c_file

            @classmethod
            def from_params(cls, params        )       :
                params.add_file_to_archive(u"c_file")
                c_file = params.pop(u"c_file")

                return cls(c_file)

        class B(object):
            def __init__(self, filename     , c)        :
                self.filename = filename
                self.c_dict = {u"here": c}

            @classmethod
            def from_params(cls, params        )       :
                params.add_file_to_archive(u"filename")

                filename = params.pop(u"filename")
                c_params = params.pop(u"c")
                c = C.from_params(c_params)

                return cls(filename, c)

        class A(object):
            def __init__(self, b)        :
                self.b = b

            @classmethod
            def from_params(cls, params        )       :
                b_params = params.pop(u"b")
                return cls(B.from_params(b_params))

        params = Params({
                u"a": {
                        u"b": {
                                u"filename": my_file,
                                u"c": {
                                        u"c_file": my_other_file
                                }
                        }
                }
        })

        # Construct ``A`` from params but then just throw it away.
        A.from_params(params.pop(u"a"))

        assert params.files_to_archive == {
                u"a.b.filename": my_file,
                u"a.b.c.c_file": my_other_file
        }

    def test_as_ordered_dict(self):
        # keyD > keyC > keyE; keyDA > keyDB; Next all other keys alphabetically
        preference_orders = [[u"keyD", u"keyC", u"keyE"], [u"keyDA", u"keyDB"]]
        params = Params({u"keyC": u"valC", u"keyB": u"valB", u"keyA": u"valA", u"keyE": u"valE",
                         u"keyD": {u"keyDB": u"valDB", u"keyDA": u"valDA"}})
        ordered_params_dict = params.as_ordered_dict(preference_orders)
        expected_ordered_params_dict = OrderedDict({u'keyD': {u'keyDA': u'valDA', u'keyDB': u'valDB'},
                                                    u'keyC': u'valC', u'keyE': u'valE',
                                                    u'keyA': u'valA', u'keyB': u'valB'})
        assert json.dumps(ordered_params_dict) == json.dumps(expected_ordered_params_dict)

    def test_to_file(self):
        # Test to_file works with or without preference orders
        params_dict = {u"keyA": u"valA", u"keyB": u"valB"}
        expected_ordered_params_dict = OrderedDict({u"keyB": u"valB", u"keyA": u"valA"})
        params = Params(params_dict)
        file_path = self.TEST_DIR / u'config.jsonnet'
        # check with preference orders
        params.to_file(file_path, [[u"keyB", u"keyA"]])
        with open(file_path, u"r") as handle:
            ordered_params_dict = OrderedDict(json.load(handle))
        assert json.dumps(expected_ordered_params_dict) == json.dumps(ordered_params_dict)
        # check without preference orders doesn't give error
        params.to_file(file_path)
