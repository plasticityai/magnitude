from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gc
import numpy as np
import os
import sys
import tempfile
import unittest

from pymagnitude import Magnitude, FeaturizerMagnitude, MagnitudeUtils
from numpy import isclose, asarray

try:
    unicode
except NameError:
    unicode = str


def _clear_mmap():
    os.system("rm -rf " + os.path.join(tempfile.gettempdir(), '*.magmmap'))
    os.system("rm -rf " + os.path.join(tempfile.gettempdir(), '*.magmmap*'))


class MagnitudeTest(unittest.TestCase):
    MAGNITUDE_PATH = ""
    MAGNITUDE_SUBWORD_PATH = ""
    MAGNITUDE_APPROX_PATH = ""

    def setUp(self):
        self.vectors = Magnitude(MagnitudeTest.MAGNITUDE_PATH,
                                 case_insensitive=True, eager=True)
        self.vectors_cs = Magnitude(MagnitudeTest.MAGNITUDE_PATH,
                                    case_insensitive=False, eager=False)
        self.vectors_sw = Magnitude(MagnitudeTest.MAGNITUDE_SUBWORD_PATH,
                                    case_insensitive=True, eager=False)
        self.vectors_approx = Magnitude(MagnitudeTest.MAGNITUDE_APPROX_PATH,
                                        case_insensitive=True, eager=False)
        self.tmp_vectors = Magnitude(MagnitudeTest.MAGNITUDE_PATH,
                                     case_insensitive=True, eager=False)
        self.concat_1 = Magnitude(MagnitudeTest.MAGNITUDE_PATH,
                                  case_insensitive=True, eager=False)
        self.concat_2 = Magnitude(MagnitudeTest.MAGNITUDE_PATH,
                                  case_insensitive=True, eager=False)
        self.concat = Magnitude(self.concat_1, self.concat_2)
        self.vectors_feat = FeaturizerMagnitude(100, case_insensitive=True)
        self.v = {
            'padding': self.tmp_vectors._padding_vector(),
            'I': self.tmp_vectors.query("I"),
            'saw': self.tmp_vectors.query("saw"),
            'a': self.tmp_vectors.query("a"),
            'cat': self.tmp_vectors.query("cat"),
            'He': self.tmp_vectors.query("He"),
            'went': self.tmp_vectors.query("went"),
            'to': self.tmp_vectors.query("to"),
            'the': self.tmp_vectors.query("the"),
            'mall': self.tmp_vectors.query("mall"),
            'blah123': self.tmp_vectors.query("blah123")
        }

    def tearDown(self):
        self.vectors.close()
        self.vectors_cs.close()
        self.vectors_sw.close()
        self.tmp_vectors.close()
        self.concat_1.close()
        self.concat_2.close()
        del self.concat
        self.vectors_feat.close()
        gc.collect()

    def test_sqlite_lib(self):
        self.assertEqual(self.vectors.sqlite_lib, 'internal')

    def test_length(self):
        self.assertEqual(len(self.vectors), 3000000)

    def test_dim(self):
        self.assertEqual(self.vectors.dim, 300)

    def test_index(self):
        self.assertTrue(isinstance(self.vectors[0][0], unicode))
        self.assertTrue(isinstance(self.vectors[0][1], np.ndarray))
        self.assertTrue(isinstance(self.vectors.index(0)[0], unicode))
        self.assertTrue(isinstance(self.vectors.index(0)[1], np.ndarray))
        self.assertTrue(isinstance(self.vectors.index(0, return_vector=False),
                                   unicode))

    def test_slice(self):
        sliced = self.vectors[0:5]
        self.assertEqual(len(sliced), 5)
        self.assertEqual(sliced[0][0], self.vectors[0][0])
        self.assertTrue(isclose(sliced[0][1], self.vectors[0][1]).all())

    def test_case_insensitive(self):
        some_keys_are_not_lower = False
        for i, (k, _) in enumerate(self.vectors):
            if i > 1000:
                break
            some_keys_are_not_lower = (some_keys_are_not_lower or
                                       k.lower() != k)
        self.assertTrue(some_keys_are_not_lower)
        self.assertTrue("QuEEn" in self.vectors)
        self.assertTrue("QUEEN" in self.vectors)
        self.assertTrue("queen" in self.vectors)
        self.assertTrue(isclose(self.vectors.query("Queen"),
                                self.vectors.query("QuEEn")).all())
        self.assertEqual(
            self.vectors.most_similar(
                "I",
                return_similarities=False)[0],
            'myself')
        self.assertEqual(
            self.vectors.most_similar(
                "i",
                return_similarities=False)[0],
            'ive')
        self.assertTrue(self.vectors.similarity("a", "A") > .9)

    def test_case_sensitive(self):
        some_keys_are_not_lower = False
        for i, (k, _) in enumerate(self.vectors_cs):
            if i > 1000:
                break
            some_keys_are_not_lower = (some_keys_are_not_lower or
                                       k.lower() != k)
        self.assertTrue(some_keys_are_not_lower)
        self.assertTrue("QuEEn" not in self.vectors_cs)
        self.assertTrue("QUEEN" in self.vectors_cs)
        self.assertTrue("queen" in self.vectors_cs)
        self.assertTrue(not isclose(self.vectors_cs.query("Queen"),
                                    self.vectors_cs.query("QuEEn")).all())
        self.assertEqual(
            self.vectors_cs.most_similar(
                "I",
                return_similarities=False)[0],
            'myself')
        self.assertEqual(self.vectors_cs.most_similar(
            "i", return_similarities=False)[0], 'ive')
        self.assertTrue(self.vectors_cs.similarity("a", "A") > .9)

    def test_iter_case_insensitive(self):
        for _ in range(2):
            for i, (k, v) in enumerate(self.vectors):
                if i > 1000:
                    break
                k2, v2 = self.vectors[i]
                self.assertEqual(k, k2)
                self.assertTrue(isclose(v[0], v2[0]))

    def test_iter_case_sensitive(self):
        for _ in range(2):
            for i, (k, v) in enumerate(self.vectors_cs):
                if i > 1000:
                    break
                k2, v2 = self.vectors_cs[i]
                self.assertEqual(k, k2)
                self.assertTrue(isclose(v[0], v2[0]))

    def test_index_case_insensitive(self):
        for _ in range(2):
            viter = iter(self.vectors)
            for i in range(len(self.vectors)):
                if i > 1000:
                    break
                k, v = next(viter)
                k2, v2 = self.vectors[i]
                self.assertEqual(k, k2)
                self.assertTrue(isclose(v[0], v2[0]))

    def test_index_case_sensitive(self):
        for _ in range(2):
            viter = iter(self.vectors_cs)
            for i in range(len(self.vectors_cs)):
                if i > 1000:
                    break
                k, v = next(viter)
                k2, v2 = self.vectors_cs[i]
                self.assertEqual(k, k2)
                self.assertTrue(isclose(v[0], v2[0]))

    def test_bounds(self):
        length = len(self.vectors)
        self.assertTrue(isinstance(self.vectors[length - 1][0], unicode))
        self.assertTrue(isinstance(self.vectors[length - 1][1], np.ndarray))

    @unittest.expectedFailure
    def test_out_of_bounds(self):
        length = len(self.vectors)
        self.assertTrue(isinstance(self.vectors[length][0], unicode))
        self.assertTrue(isinstance(self.vectors[length][1], np.ndarray))

    def test_contains(self):
        self.assertTrue("cat" in self.vectors)

    def test_contains_false(self):
        self.assertTrue("blah123" not in self.vectors)

    def test_special_characters(self):
        self.assertTrue("Wilkes-Barre/Scranton" in self.vectors)
        self.assertTrue("out-of-vocabulary" not in self.vectors)
        self.assertTrue('quotation"s' not in self.vectors)
        self.assertTrue("quotation's" not in self.vectors)
        self.assertTrue("colon;s" not in self.vectors)
        self.assertTrue("sh**" not in self.vectors)
        self.assertTrue("'s" not in self.vectors_cs)
        self.assertTrue('"s' not in self.vectors)
        self.assertEqual(self.vectors.query("cat").shape,
                         self.vectors.query("Wilkes-Barre/Scranton").shape)
        self.assertEqual(self.vectors.query("cat").shape,
                         self.vectors.query("out-of-vocabulary").shape)
        self.assertEqual(self.vectors.query("cat").shape,
                         self.vectors.query('quotation"s').shape)
        self.assertEqual(self.vectors.query("cat").shape,
                         self.vectors.query("quotation's").shape)
        self.assertEqual(self.vectors.query("cat").shape,
                         self.vectors.query("colon;s").shape)
        self.assertEqual(self.vectors.query("cat").shape,
                         self.vectors.query("sh**").shape)
        self.assertEqual(self.vectors.query("cat").shape,
                         self.vectors_cs.query("'s").shape)
        self.assertEqual(self.vectors.query("cat").shape,
                         self.vectors.query('"s').shape)

    def test_oov_dim(self):
        self.assertEqual(self.vectors.query("*<<<<").shape,
                         self.vectors.query("cat").shape)

    def test_oov_subword_dim(self):
        self.assertEqual(self.vectors_sw.query("*<<<<").shape,
                         self.vectors_sw.query("cat").shape)

    def test_oov_dim_placeholders(self):
        self.vectors_placeholders = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            placeholders=5,
            case_insensitive=True,
            eager=False)
        self.assertEqual(self.vectors_placeholders.query("*<<<<").shape,
                         self.vectors_placeholders.query("cat").shape)
        self.assertTrue(isclose(self.vectors.query("*<<<<")[0],
                                self.vectors_placeholders.query("*<<<<")[0]))
        self.vectors_placeholders.close()

    def test_oov_subword_dim_placeholders(self):
        self.vectors_placeholders = Magnitude(
            MagnitudeTest.MAGNITUDE_SUBWORD_PATH, placeholders=5,
            case_insensitive=True, eager=False)
        self.assertEqual(self.vectors_placeholders.query("*<<<<").shape,
                         self.vectors_placeholders.query("cat").shape)
        self.assertTrue(isclose(self.vectors.query("*<<<<")[0],
                                self.vectors_placeholders.query("*<<<<")[0]))
        self.vectors_placeholders.close()

    def test_oov_unit_norm(self):
        self.assertTrue(isclose(np.linalg.norm(self.vectors.query("*<<<<<")),
                                1.0))

    def test_oov_subword_unit_norm(self):
        self.assertTrue(
            isclose(
                np.linalg.norm(
                    self.vectors_sw.query("*<<<<<")),
                1.0))

    def test_ngram_oov_closeness(self):
        self.assertTrue(self.vectors.similarity("uberx", "uberxl") > .7)
        self.assertTrue(self.vectors.similarity("uberx", "veryrandom") < .7)
        self.assertTrue(self.vectors.similarity("veryrandom",
                                                "veryrandom") > .7)

    def test_ngram_oov_subword_closeness(self):
        self.assertTrue(self.vectors_sw.similarity("uberx", "uberxl") > .7)
        self.assertTrue(self.vectors_sw.similarity("uberx", "uber") > .7)
        self.assertTrue(self.vectors_sw.similarity("uberxl", "uber") > .7)
        self.assertTrue(self.vectors_sw.similarity("discriminatoryy",
                                                   "discriminatory") > .7)
        self.assertTrue(self.vectors_sw.similarity("discriminatoryy",
                                                   "discriminnatory") > .8)
        self.assertTrue(self.vectors_sw.similarity("uberx", "veryrandom") <
                        .7)
        self.assertTrue(self.vectors_sw.similarity("veryrandom",
                                                   "veryrandom") > .7)
        self.assertTrue(self.vectors_sw.similarity("hiiiiiiiii",
                                                   "hi") > .7)
        self.assertTrue(self.vectors_sw.similarity("heeeeeeeey",
                                                   "hey") > .7)
        self.assertTrue(self.vectors_sw.similarity("heyyyyyyyyyy",
                                                   "hey") > .7)
        self.assertTrue(self.vectors_sw.similarity("faaaaaate",
                                                   "fate") > .65)

    def test_oov_values(self):
        self.vectors_oov_1 = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            case_insensitive=True,
            ngram_oov=False,
            eager=False)
        self.vectors_oov_2 = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            case_insensitive=True,
            ngram_oov=False,
            eager=False)

        self.assertTrue(isclose(self.vectors_oov_1.query("*<")[0],
                                -0.0759614511397))
        self.assertTrue(isclose(self.vectors_oov_1.query("*<<")[0],
                                0.00742723997271))
        self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<")[0],
                                -0.0372075283555))
        self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<<")[0],
                                -0.0201727917272))
        self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<<<")[0],
                                -0.0475993225776))
        self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<<<<")[0],
                                0.0129938352266))
        self.assertTrue(isclose(self.vectors_oov_2.query("*<")[0],
                                -0.0759614511397))
        self.assertTrue(isclose(self.vectors_oov_2.query("*<<")[0],
                                0.00742723997271))
        self.assertTrue(isclose(self.vectors_oov_2.query("*<<<<")[0],
                                -0.0372075283555))
        self.assertTrue(isclose(self.vectors_oov_2.query("*<<<<<")[0],
                                -0.0201727917272))
        self.assertTrue(isclose(self.vectors_oov_2.query("*<<<<<<")[0],
                                -0.0475993225776))
        self.assertTrue(isclose(self.vectors_oov_2.query("*<<<<<<<")[0],
                                0.0129938352266))

        self.vectors_oov_1.close()
        self.vectors_oov_2.close()

    def test_oov_subword_values(self):
        self.vectors_oov_1 = Magnitude(
            MagnitudeTest.MAGNITUDE_SUBWORD_PATH,
            case_insensitive=True,
            ngram_oov=False,
            eager=False)
        self.vectors_oov_2 = Magnitude(
            MagnitudeTest.MAGNITUDE_SUBWORD_PATH,
            case_insensitive=True,
            ngram_oov=False,
            eager=False)

        self.assertTrue(isclose(self.vectors_oov_1.query("discriminatoryy")[0],
                                -0.059116619334669426))
        self.assertTrue(isclose(self.vectors_oov_1.query("*<")[0],
                                -0.0759614511397))
        self.assertTrue(isclose(self.vectors_oov_1.query("*<<")[0],
                                0.00742723997271))
        self.assertTrue(isclose(self.vectors_oov_1.query("uberx")[0],
                                0.0952671681336))
        self.assertTrue(isclose(self.vectors_oov_1.query("misssipi")[0],
                                0.0577835297955))
        self.assertTrue(isclose(self.vectors_oov_2.query("discriminatoryy")[0],
                                -0.059116619334669426))
        self.assertTrue(isclose(self.vectors_oov_2.query("*<")[0],
                                -0.0759614511397))
        self.assertTrue(isclose(self.vectors_oov_2.query("*<<")[0],
                                0.00742723997271))
        self.assertTrue(isclose(self.vectors_oov_2.query("uberx")[0],
                                0.0952671681336))
        self.assertTrue(isclose(self.vectors_oov_2.query("misssipi")[0],
                                0.0577835297955))

        self.vectors_oov_1.close()
        self.vectors_oov_2.close()

    def test_oov_stability(self):
        self.vectors_oov_1 = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            case_insensitive=True,
            ngram_oov=False,
            eager=False)
        self.vectors_oov_2 = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            case_insensitive=True,
            ngram_oov=False,
            eager=False)

        for i in range(5):
            self.assertTrue(isclose(self.vectors_oov_1.query("*<"),
                                    self.vectors_oov_2.query("*<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<"),
                                    self.vectors_oov_2.query("*<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<"),
                                    self.vectors_oov_2.query("*<<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<"),
                                    self.vectors_oov_2.query("*<<<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<<"),
                                    self.vectors_oov_2.query("*<<<<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<<<"),
                                    self.vectors_oov_2.query("*<<<<<<")).all())
            self.assertTrue(
                isclose(
                    self.vectors_oov_1.query("*<<<<<<<"),
                    self.vectors_oov_2.query("*<<<<<<<")).all())

        self.vectors_oov_1.close()
        self.vectors_oov_2.close()

    def test_ngram_oov_stability(self):
        self.vectors_oov_1 = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            case_insensitive=True,
            ngram_oov=True,
            eager=False)
        self.vectors_oov_2 = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            case_insensitive=True,
            ngram_oov=True,
            eager=False)

        for i in range(5):
            self.assertTrue(isclose(self.vectors_oov_1.query("*<"),
                                    self.vectors_oov_2.query("*<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<"),
                                    self.vectors_oov_2.query("*<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<"),
                                    self.vectors_oov_2.query("*<<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<"),
                                    self.vectors_oov_2.query("*<<<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<<"),
                                    self.vectors_oov_2.query("*<<<<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<<<"),
                                    self.vectors_oov_2.query("*<<<<<<")).all())
            self.assertTrue(
                isclose(
                    self.vectors_oov_1.query("*<<<<<<<"),
                    self.vectors_oov_2.query("*<<<<<<<")).all())

        self.vectors_oov_1.close()
        self.vectors_oov_2.close()

    def test_ngram_oov_subword_stability(self):
        self.vectors_oov_1 = Magnitude(MagnitudeTest.MAGNITUDE_SUBWORD_PATH,
                                       case_insensitive=True, eager=False)
        self.vectors_oov_2 = Magnitude(MagnitudeTest.MAGNITUDE_SUBWORD_PATH,
                                       case_insensitive=True, eager=False)

        for i in range(5):
            self.assertTrue(isclose(self.vectors_oov_1.query("*<"),
                                    self.vectors_oov_2.query("*<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<"),
                                    self.vectors_oov_2.query("*<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<"),
                                    self.vectors_oov_2.query("*<<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<"),
                                    self.vectors_oov_2.query("*<<<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<<"),
                                    self.vectors_oov_2.query("*<<<<<")).all())
            self.assertTrue(isclose(self.vectors_oov_1.query("*<<<<<<"),
                                    self.vectors_oov_2.query("*<<<<<<")).all())
            self.assertTrue(
                isclose(
                    self.vectors_oov_1.query("*<<<<<<<"),
                    self.vectors_oov_2.query("*<<<<<<<")).all())

        self.vectors_oov_1.close()
        self.vectors_oov_2.close()

    def test_oov_subword_long_key(self):
        self.vectors_sw.query('ab' * 1026)
        # Previous line should not fail
        self.assertTrue(True)

    def test_lang_english_oov_stem(self):
        self.assertEqual(self.vectors._oov_stem('rejumping'), 'jump')
        self.assertEqual(self.vectors._oov_stem(
            'pre-reuberification-ing'), 'uber')
        self.assertTrue(self.vectors_sw.similarity("houuuuuuse", "house") > .67)
        self.assertTrue(isclose(self.vectors_sw.query(
            "houuuuuuse")[0], -0.007254118679147659))
        self.assertTrue(
            self.vectors_sw.similarity(
                "skillllllll",
                "skill") > .58)
        self.assertTrue(isclose(self.vectors_sw.query(
            "skillllllll")[0], 0.0039352450099857696))
        self.assertTrue(
            self.vectors_sw.similarity(
                "uberification",
                "uber") > .7)
        self.assertTrue(
            isclose(
                self.vectors_sw.query("uberification")[0],
                0.033199077449376516))
        self.assertTrue(
            self.vectors_sw.similarity(
                "uberificatttttioooooonn",
                "uber") > .7)
        self.assertTrue(
            isclose(
                self.vectors_sw.query("uberificatttttioooooonn")[0],
                0.0527393434944338))
        self.assertTrue(self.vectors_sw.similarity("sjump", "jump") > .68)
        self.assertTrue(
            isclose(
                self.vectors_sw.query("sjump")[0],
                0.04112182276959207))
        self.assertTrue(self.vectors_sw.similarity("sjumping", "jumping") > .7)
        self.assertTrue(
            isclose(
                self.vectors_sw.query("sjumping")[0],
                0.048280197411183244))
        self.assertTrue(
            self.vectors_sw.similarity(
                "sjumpinnnnnnnnng",
                "jump") > .7)
        self.assertTrue(
            isclose(
                self.vectors_sw.query("sjumpinnnnnnnnng")[0],
                0.011190570778487749))

    def test_lang_none_oov_stem(self):
        self.vectors_l = Magnitude(MagnitudeTest.MAGNITUDE_PATH, language=None)
        self.assertEqual(self.vectors_l._oov_stem('rejumping'), 'rejumping')
        self.assertEqual(
            self.vectors_l._oov_stem('reuberificationing'),
            'reuberificationing')
        self.vectors_l.close()

    def test_placeholders(self):
        self.vectors_placeholders = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            case_insensitive=True,
            placeholders=5,
            eager=False)
        self.assertEqual(self.vectors_placeholders.query("cat").shape, (305,))
        self.assertEqual(self.vectors_placeholders.query("cat")[0],
                         self.vectors.query("cat")[0])
        self.vectors_placeholders.close()

    def test_numpy(self):
        self.assertTrue(isinstance(self.vectors.query("cat"), np.ndarray))

    def test_list(self):
        self.vectors_list = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            case_insensitive=True,
            use_numpy=False,
            eager=False)
        self.assertTrue(isinstance(self.vectors_list.query("cat"), list))
        self.vectors_list.close()

    def test_repeated_single(self):
        q = "cat"
        result = self.vectors.query(q)
        result_2 = self.vectors.query(q)
        self.assertTrue(isclose(result, result_2).all())

    def test_repeated_multiple(self):
        q = ["I", "saw", "a", "cat"]
        result = self.vectors.query(q)
        result_2 = self.vectors.query(q)
        self.assertTrue(isclose(result, result_2).all())
        q = [["I", "saw", "a", "cat"], ["He", "went", "to", "the", "mall"]]
        result = self.vectors.query(q)
        result_2 = self.vectors.query(q)
        self.assertTrue(isclose(result, result_2).all())

    def test_multiple(self):
        q = [["I", "saw", "a", "cat"], ["He", "went", "to", "the", "mall"]]
        result = self.vectors.query(q)
        self.assertEqual(result.shape, (2, 5, self.vectors.dim))
        self.assertTrue(isclose(result[0][0], self.v['I']).all())
        self.assertTrue(isclose(result[0][1], self.v['saw']).all())
        self.assertTrue(isclose(result[0][2], self.v['a']).all())
        self.assertTrue(isclose(result[0][3], self.v['cat']).all())
        self.assertTrue(isclose(result[0][4], self.v['padding']).all())
        self.assertTrue(isclose(result[1][0], self.v['He']).all())
        self.assertTrue(isclose(result[1][1], self.v['went']).all())
        self.assertTrue(isclose(result[1][2], self.v['to']).all())
        self.assertTrue(isclose(result[1][3], self.v['the']).all())
        self.assertTrue(isclose(result[1][4], self.v['mall']).all())
        return result

    def test_pad_to_length_right_truncate_none(self):
        q = [["I", "saw", "a", "cat"], ["He", "went", "to", "the", "mall"]]
        result = self.vectors.query(q, pad_to_length=6)
        self.assertEqual(result.shape, (2, 6, self.vectors.dim))
        self.assertTrue(isclose(result[0][0], self.v['I']).all())
        self.assertTrue(isclose(result[0][1], self.v['saw']).all())
        self.assertTrue(isclose(result[0][2], self.v['a']).all())
        self.assertTrue(isclose(result[0][3], self.v['cat']).all())
        self.assertTrue(isclose(result[0][4], self.v['padding']).all())
        self.assertTrue(isclose(result[0][5], self.v['padding']).all())
        self.assertTrue(isclose(result[1][0], self.v['He']).all())
        self.assertTrue(isclose(result[1][1], self.v['went']).all())
        self.assertTrue(isclose(result[1][2], self.v['to']).all())
        self.assertTrue(isclose(result[1][3], self.v['the']).all())
        self.assertTrue(isclose(result[1][4], self.v['mall']).all())
        self.assertTrue(isclose(result[1][5], self.v['padding']).all())
        return result

    def test_pad_to_length_truncate_none(self):
        q = [["I", "saw", "a", "cat"], ["He", "went", "to", "the", "mall"]]
        result = self.vectors.query(q, pad_to_length=6)
        self.assertEqual(result.shape, (2, 6, self.vectors.dim))
        self.assertTrue(isclose(result[0][0], self.v['I']).all())
        self.assertTrue(isclose(result[0][1], self.v['saw']).all())
        self.assertTrue(isclose(result[0][2], self.v['a']).all())
        self.assertTrue(isclose(result[0][3], self.v['cat']).all())
        self.assertTrue(isclose(result[0][4], self.v['padding']).all())
        self.assertTrue(isclose(result[0][5], self.v['padding']).all())
        self.assertTrue(isclose(result[1][0], self.v['He']).all())
        self.assertTrue(isclose(result[1][1], self.v['went']).all())
        self.assertTrue(isclose(result[1][2], self.v['to']).all())
        self.assertTrue(isclose(result[1][3], self.v['the']).all())
        self.assertTrue(isclose(result[1][4], self.v['mall']).all())
        self.assertTrue(isclose(result[1][5], self.v['padding']).all())
        return result

    def test_pad_to_length_left_truncate_none(self):
        q = [["I", "saw", "a", "cat"], ["He", "went", "to", "the", "mall"]]
        result = self.vectors.query(q, pad_to_length=6, pad_left=True)
        self.assertEqual(result.shape, (2, 6, self.vectors.dim))
        self.assertTrue(isclose(result[0][0], self.v['padding']).all())
        self.assertTrue(isclose(result[0][1], self.v['padding']).all())
        self.assertTrue(isclose(result[0][2], self.v['I']).all())
        self.assertTrue(isclose(result[0][3], self.v['saw']).all())
        self.assertTrue(isclose(result[0][4], self.v['a']).all())
        self.assertTrue(isclose(result[0][5], self.v['cat']).all())
        self.assertTrue(isclose(result[1][0], self.v['padding']).all())
        self.assertTrue(isclose(result[1][1], self.v['He']).all())
        self.assertTrue(isclose(result[1][2], self.v['went']).all())
        self.assertTrue(isclose(result[1][3], self.v['to']).all())
        self.assertTrue(isclose(result[1][4], self.v['the']).all())
        self.assertTrue(isclose(result[1][5], self.v['mall']).all())
        return result

    def test_pad_to_length_truncate_right(self):
        q = [["I", "saw", "a", "cat"], ["He", "went", "to", "the", "mall"]]
        result = self.vectors.query(q, pad_to_length=3)
        self.assertEqual(result.shape, (2, 3, self.vectors.dim))
        self.assertTrue(isclose(result[0][0], self.v['I']).all())
        self.assertTrue(isclose(result[0][1], self.v['saw']).all())
        self.assertTrue(isclose(result[0][2], self.v['a']).all())
        self.assertTrue(isclose(result[1][0], self.v['He']).all())
        self.assertTrue(isclose(result[1][1], self.v['went']).all())
        self.assertTrue(isclose(result[1][2], self.v['to']).all())
        return result

    def test_pad_to_length_truncate_left(self):
        q = [["I", "saw", "a", "cat"], ["He", "went", "to", "the", "mall"]]
        result = self.vectors.query(q, pad_to_length=3, truncate_left=True)
        self.assertEqual(result.shape, (2, 3, self.vectors.dim))
        self.assertTrue(isclose(result[0][0], self.v['saw']).all())
        self.assertTrue(isclose(result[0][1], self.v['a']).all())
        self.assertTrue(isclose(result[0][2], self.v['cat']).all())
        self.assertTrue(isclose(result[1][0], self.v['to']).all())
        self.assertTrue(isclose(result[1][1], self.v['the']).all())
        self.assertTrue(isclose(result[1][2], self.v['mall']).all())
        return result

    def test_list_multiple(self):
        self.vectors_list = Magnitude(
            MagnitudeTest.MAGNITUDE_PATH,
            case_insensitive=True,
            use_numpy=False,
            eager=False)
        q = [["I", "saw", "a", "cat"], ["He", "went", "to", "the", "mall"]]
        self.assertTrue(isinstance(self.vectors_list.query(q[0]), list))
        self.assertTrue(isclose(self.vectors.query(q[0]),
                                asarray(self.vectors_list.query(q[0]))).all())
        self.assertTrue(isinstance(self.vectors_list.query(q), list))
        self.assertTrue(isclose(self.vectors.query(q),
                                asarray(self.vectors_list.query(q))).all())
        self.vectors_list.close()

    def test_concat(self):
        q = "cat"
        result = self.concat.query(q)
        self.assertEqual(result.shape, (self.vectors.dim * 2,))
        self.assertTrue(isclose(result[0:300], self.v['cat']).all())
        self.assertTrue(isclose(result[300:600], self.v['cat']).all())

    def test_concat_multiple(self):
        q = ["I", "saw"]
        result = self.concat.query(q)
        self.assertEqual(result.shape, (2, self.vectors.dim * 2,))
        self.assertTrue(isclose(result[0][0:300], self.v['I']).all())
        self.assertTrue(isclose(result[0][300:600], self.v['I']).all())
        self.assertTrue(isclose(result[1][0:300], self.v['saw']).all())
        self.assertTrue(isclose(result[1][300:600], self.v['saw']).all())

    def test_concat_multiple_2(self):
        q = [["I", "saw"], ["He", "went"]]
        result = self.concat.query(q)
        self.assertEqual(result.shape, (2, 2, self.vectors.dim * 2,))
        self.assertTrue(isclose(result[0][0][0:300], self.v['I']).all())
        self.assertTrue(isclose(result[0][0][300:600], self.v['I']).all())
        self.assertTrue(isclose(result[0][1][0:300], self.v['saw']).all())
        self.assertTrue(isclose(result[0][1][300:600], self.v['saw']).all())
        self.assertTrue(isclose(result[1][0][0:300], self.v['He']).all())
        self.assertTrue(isclose(result[1][0][300:600], self.v['He']).all())
        self.assertTrue(isclose(result[1][1][0:300], self.v['went']).all())
        self.assertTrue(isclose(result[1][1][300:600], self.v['went']).all())

    def test_concat_specific(self):
        q = ("cat", "mall")
        result = self.concat.query(q)
        self.assertEqual(result.shape, (self.vectors.dim * 2,))
        self.assertTrue(isclose(result[0:300], self.v['cat']).all())
        self.assertTrue(isclose(result[300:600], self.v['mall']).all())

    def test_concat_multiple_specific(self):
        q = [("I", "He"), ("saw", "went")]
        result = self.concat.query(q)
        self.assertEqual(result.shape, (2, self.vectors.dim * 2,))
        self.assertTrue(isclose(result[0][0:300], self.v['I']).all())
        self.assertTrue(isclose(result[0][300:600], self.v['He']).all())
        self.assertTrue(isclose(result[1][0:300], self.v['saw']).all())
        self.assertTrue(isclose(result[1][300:600], self.v['went']).all())

    def test_concat_multiple_2_specific(self):
        q = [[("I", "He"), ("saw", "went")], [("He", "I"), ("went", "saw")]]
        result = self.concat.query(q)
        self.assertEqual(result.shape, (2, 2, self.vectors.dim * 2,))
        self.assertTrue(isclose(result[0][0][0:300], self.v['I']).all())
        self.assertTrue(isclose(result[0][0][300:600], self.v['He']).all())
        self.assertTrue(isclose(result[0][1][0:300], self.v['saw']).all())
        self.assertTrue(isclose(result[0][1][300:600], self.v['went']).all())
        self.assertTrue(isclose(result[1][0][0:300], self.v['He']).all())
        self.assertTrue(isclose(result[1][0][300:600], self.v['I']).all())
        self.assertTrue(isclose(result[1][1][0:300], self.v['went']).all())
        self.assertTrue(isclose(result[1][1][300:600], self.v['saw']).all())

    def test_distance(self):
        self.assertTrue(isclose(self.vectors.distance("cat", "dog"),
                                0.69145405))

    def test_distance_multiple(self):
        self.assertTrue(isclose(self.vectors.distance("cat", ["cats", "dog"]),
                                [0.61654216, 0.69145405]).all())

    def test_similarity(self):
        self.assertTrue(isclose(self.vectors.similarity("cat", "dog"),
                                0.7609457089782209))

    def test_similarity_multiple(self):
        self.assertTrue(
            isclose(
                self.vectors.similarity(
                    "cat", [
                        "cats", "dog"]), [
                    0.8099378824686305, 0.7609457089782209]).all())

    def test_most_similar_to_given(self):
        self.assertEqual(self.vectors.most_similar_to_given(
            "cat", ["dog", "television", "laptop"]), "dog")
        self.assertEqual(self.vectors.most_similar_to_given(
            "cat", ["television", "dog", "laptop"]), "dog")
        self.assertEqual(self.vectors.most_similar_to_given(
            "cat", ["television", "laptop", "dog"]), "dog")

    def test_doesnt_match(self):
        self.assertEqual(self.vectors.doesnt_match(
            ["breakfast", "cereal", "lunch", "dinner"]), "cereal")
        self.assertEqual(self.vectors.doesnt_match(
            ["breakfast", "lunch", "cereal", "dinner"]), "cereal")
        self.assertEqual(self.vectors.doesnt_match(
            ["breakfast", "lunch", "dinner", "cereal"]), "cereal")

    def test_most_similar_case_insensitive(self):
        keys = [s[0] for s in self.vectors.most_similar("queen",
                                                        topn=5)]
        similarities = [s[1] for s in self.vectors.most_similar("queen",
                                                                topn=5)]
        self.assertTrue(isclose(asarray(similarities),
                                asarray([0.7399442791938782,
                                         0.7070531845092773,
                                         0.6510956287384033,
                                         0.6383601427078247,
                                         0.6357027292251587
                                         ]), atol=.02).all())
        self.assertEqual(keys,
                         [u'queens',
                          u'princess',
                          u'king',
                          u'monarch',
                          u'very_pampered_McElhatton'
                          ])

    def test_most_similar(self):
        keys = [s[0] for s in self.vectors_cs.most_similar("queen")]
        similarities = [s[1] for s in self.vectors_cs.most_similar("queen")]
        self.assertTrue(isclose(asarray(similarities),
                                asarray([0.7399442791938782,
                                         0.7070531845092773,
                                         0.6510956287384033,
                                         0.6383601427078247,
                                         0.6357027292251587,
                                         0.6163408160209656,
                                         0.6060680150985718,
                                         0.5923796892166138,
                                         0.5908075571060181,
                                         0.5637184381484985
                                         ]), atol=.02).all())
        self.assertEqual(keys,
                         [u'queens',
                          u'princess',
                          u'king',
                          u'monarch',
                          u'very_pampered_McElhatton',
                          u'Queen',
                          u'NYC_anglophiles_aflutter',
                          u'Queen_Consort',
                          u'princesses',
                          u'royal',
                          ])

    def test_most_similar_no_similarities(self):
        keys = self.vectors_cs.most_similar("queen",
                                            return_similarities=False)
        self.assertEqual(keys,
                         [u'queens',
                          u'princess',
                          u'king',
                          u'monarch',
                          u'very_pampered_McElhatton',
                          u'Queen',
                          u'NYC_anglophiles_aflutter',
                          u'Queen_Consort',
                          u'princesses',
                          u'royal',
                          ])

    def test_most_similar_top_5(self):
        keys = [s[0] for s in self.vectors_cs.most_similar("queen", topn=5)]
        similarities = [s[1] for s in self.vectors_cs.most_similar("queen",
                                                                   topn=5)]
        self.assertTrue(isclose(asarray(similarities),
                                asarray([0.7399442791938782,
                                         0.7070531845092773,
                                         0.6510956287384033,
                                         0.6383601427078247,
                                         0.6357027292251587
                                         ]), atol=.02).all())
        self.assertEqual(keys,
                         [u'queens',
                          u'princess',
                          u'king',
                          u'monarch',
                          u'very_pampered_McElhatton'
                          ])

    def test_most_similar_min_similarity(self):
        keys = [s[0] for s in self.vectors_cs.most_similar("queen",
                                                           min_similarity=.63)]
        similarities = [
            s[1] for s in self.vectors_cs.most_similar(
                "queen", min_similarity=.63)]
        self.assertTrue(isclose(asarray(similarities),
                                asarray([0.7399442791938782,
                                         0.7070531845092773,
                                         0.6510956287384033,
                                         0.6383601427078247,
                                         0.6357027292251587
                                         ]), atol=.02).all())
        self.assertEqual(keys,
                         [u'queens',
                          u'princess',
                          u'king',
                          u'monarch',
                          u'very_pampered_McElhatton'
                          ])

    def test_most_similar_analogy(self):
        keys = [s[0] for s in self.vectors_cs.most_similar(
            positive=["king", "woman"], negative=["man"])]
        similarities = [s[1] for s in self.vectors_cs.most_similar(
            positive=["king", "woman"], negative=["man"])]
        self.assertTrue(isclose(asarray(similarities),
                                asarray([0.7118192315101624,
                                         0.6189674139022827,
                                         0.5902431011199951,
                                         0.549946129322052,
                                         0.5377321243286133,
                                         0.5236844420433044,
                                         0.5235944986343384,
                                         0.518113374710083,
                                         0.5098593831062317,
                                         0.5087411403656006
                                         ]), atol=.02).all())
        self.assertEqual(keys,
                         [u'queen',
                          u'monarch',
                          u'princess',
                          u'crown_prince',
                          u'prince',
                          u'kings',
                          u'Queen_Consort',
                          u'queens',
                          u'sultan',
                          u'monarchy'
                          ])

    def test_most_similar_cosmul_analogy(self):
        keys = [s[0] for s in self.vectors_cs.most_similar_cosmul(
            positive=["king", "woman"], negative=["man"])]
        similarities = [s[1] for s in self.vectors_cs.most_similar_cosmul(
            positive=["king", "woman"], negative=["man"])]
        self.assertTrue(isclose(asarray(similarities),
                                asarray([0.9314123392105103,
                                         0.858533501625061,
                                         0.8476565480232239,
                                         0.8150269985198975,
                                         0.809981644153595,
                                         0.8089977502822876,
                                         0.8027306795120239,
                                         0.801961362361908,
                                         0.8009798526763916,
                                         0.7958389520645142
                                         ]), atol=.02).all())
        self.assertEqual(keys,
                         [u'queen',
                          u'monarch',
                          u'princess',
                          u'Queen_Consort',
                          u'queens',
                          u'crown_prince',
                          u'royal_palace',
                          u'monarchy',
                          u'prince',
                          u'empress'
                          ])

    def test_most_similar_cosmul_min_similarity_analogy(self):
        keys = [s[0] for s in self.vectors_cs.most_similar_cosmul(
            positive=["king", "woman"], negative=["man"], min_similarity=.81)]
        similarities = [s[1] for s in self.vectors_cs.most_similar_cosmul(
            positive=["king", "woman"], negative=["man"], min_similarity=.81)]
        self.assertTrue(isclose(asarray(similarities),
                                asarray([0.9314123392105103,
                                         0.858533501625061,
                                         0.8476565480232239,
                                         0.8150269985198975
                                         ]), atol=.02).all())
        self.assertEqual(keys,
                         [u'queen',
                          u'monarch',
                          u'princess',
                          u'Queen_Consort'
                          ])

    def test_closer_than(self):
        self.assertEqual(self.vectors.closer_than("cat", "dog"), ["cats"])

    def test_most_similar_approx(self):
        keys = [s[0] for s in self.vectors_approx.most_similar_approx(
            "queen", topn=15)]
        similarities = [s[1] for s in self.vectors_approx.most_similar_approx(
            "queen", topn=15)]
        self.assertEqual(len(keys), 15)
        self.assertTrue(similarities[0] > .7 and similarities[-1] > .5)

    @unittest.expectedFailure
    def test_most_similar_approx_failure(self):
        self.vectors.most_similar_approx("queen", topn=15)

    def test_most_similar_approx_low_effort(self):
        keys = [s[0] for s in self.vectors_approx.most_similar_approx(
            "queen", topn=15, effort=.1)]
        self.assertEqual(len(keys), 15)
        self.assertEqual(keys[0], "princess")

    def test_most_similar_analogy_approx(self):
        keys = [s[0] for s in self.vectors_approx.most_similar_approx(
            positive=["king", "woman"], negative=["man"], topn=15)]
        self.assertEqual(keys[0], "queen")

    def test_feat_length(self):
        self.vectors_feat_2 = FeaturizerMagnitude(1000, case_insensitive=True)
        self.assertEqual(self.vectors_feat.dim, 4)
        self.assertEqual(self.vectors_feat_2.dim, 5)
        self.vectors_feat_2.close()

    def test_feat_stability(self):
        self.vectors_feat_2 = FeaturizerMagnitude(100, case_insensitive=True)
        self.assertTrue(isclose(self.vectors_feat.query("VBG"),
                                self.vectors_feat_2.query("VBG")).all())
        self.assertTrue(isclose(self.vectors_feat.query("PRP"),
                                self.vectors_feat_2.query("PRP")).all())
        self.vectors_feat_2.close()

    def test_feat_values(self):
        self.assertTrue(isclose(self.vectors_feat.query("VBG")[0],
                                0.490634876828))
        self.assertTrue(isclose(self.vectors_feat.query("PRP")[0],
                                0.463890807802))
        self.assertTrue(isclose(self.vectors_feat.query(5)[0],
                                -0.750681075834))
        self.assertTrue(isclose(self.vectors_feat.query(5)[-1],
                                1.46936807866e-38))

    def test_batchify(self):
        X = [0, 1, 2, 3, 4, 5]  # noqa: N806
        y = [0, 0, 1, 1, 0, 1]
        batch_gen = MagnitudeUtils.batchify(X, y, 2)
        X_batch, y_batch = next(batch_gen)  # noqa: N806
        self.assertEqual(X_batch, [0, 1])
        self.assertEqual(y_batch, [0, 0])
        X_batch, y_batch = next(batch_gen)  # noqa: N806
        self.assertEqual(X_batch, [2, 3])
        self.assertEqual(y_batch, [1, 1])
        X_batch, y_batch = next(batch_gen)  # noqa: N806
        self.assertEqual(X_batch, [4, 5])
        self.assertEqual(y_batch, [0, 1])
        X_batch, y_batch = next(batch_gen)  # noqa: N806
        self.assertEqual(X_batch, [0, 1])
        self.assertEqual(y_batch, [0, 0])
        X = [0, 1, 2]  # noqa: N806
        y = [0, 0, 1]
        batch_gen = MagnitudeUtils.batchify(X, y, 2)
        X_batch, y_batch = next(batch_gen)  # noqa: N806
        self.assertEqual(X_batch, [0, 1])
        self.assertEqual(y_batch, [0, 0])
        X_batch, y_batch = next(batch_gen)  # noqa: N806
        self.assertEqual(X_batch, [2])
        self.assertEqual(y_batch, [1])

    def test_class_encoding(self):
        add_class, class_to_int, int_to_class = MagnitudeUtils.class_encoding()
        self.assertEqual(add_class('cat'), 0)
        self.assertEqual(add_class('dog'), 1)
        self.assertEqual(add_class('dog'), 1)
        self.assertEqual(add_class('dog'), 1)
        self.assertEqual(add_class('cat'), 0)
        self.assertEqual(class_to_int('dog'), 1)
        self.assertEqual(class_to_int('cat'), 0)
        self.assertEqual(int_to_class(1), 'dog')
        self.assertEqual(int_to_class(0), 'cat')

    def test_to_categorical(self):
        y = [1, 5, 1, 1, 2, 4, 1, 3, 1, 3, 5, 4]
        self.assertTrue(isclose(
            MagnitudeUtils.to_categorical(y),
            [[0., 1., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 1.],
             [0., 1., 0., 0., 0., 0.],
             [0., 1., 0., 0., 0., 0.],
             [0., 0., 1., 0., 0., 0.],
             [0., 0., 0., 0., 1., 0.],
             [0., 1., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0.],
             [0., 1., 0., 0., 0., 0.],
             [0., 0., 0., 1., 0., 0.],
             [0., 0., 0., 0., 0., 1.],
             [0., 0., 0., 0., 1., 0.]]
        ).all())

    def test_from_categorical(self):
        y_c = [[0., 1., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 1.]]
        self.assertTrue(isclose(
            MagnitudeUtils.from_categorical(y_c),
            [1., 5.]
        ).all())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        help="path to Google News magnitude file",
        required=True,
        type=str)
    parser.add_argument(
        "-s", "--subword-input",
        help="path to Google News magnitude file with subword information",
        required=True,
        type=str)
    parser.add_argument(
        "-a", "--approx-input",
        help="path to Google News magnitude file with a approximate nearest \
         neighbors index",
        required=True,
        type=str)
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    MagnitudeTest.MAGNITUDE_PATH = args.input
    MagnitudeTest.MAGNITUDE_SUBWORD_PATH = args.subword_input
    MagnitudeTest.MAGNITUDE_APPROX_PATH = args.approx_input
    _clear_mmap()
    unittest.main(argv=[sys.argv[0]] + args.unittest_args)
