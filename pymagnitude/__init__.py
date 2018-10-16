from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import difflib
import gc
import os
import re
import sys
import hashlib
import heapq
import lz4.frame
import math
import operator
import tempfile
import threading
import xxhash
import numpy as np
import uuid

from annoy import AnnoyIndex
from fasteners import InterProcessLock
from itertools import cycle, islice, chain, tee
from numbers import Number
from time import sleep

from pymagnitude.converter_shared import DEFAULT_NGRAM_END
from pymagnitude.converter_shared import BOW, EOW
from pymagnitude.converter_shared import CONVERTER_VERSION
from pymagnitude.converter_shared import fast_md5_file
from pymagnitude.converter_shared import char_ngrams
from pymagnitude.converter_shared import norm_matrix
from pymagnitude.third_party.repoze.lru import lru_cache

try:
    from itertools import imap
except ImportError:
    imap = map
try:
    from itertools import izip
except ImportError:
    izip = zip
try:
    unicode
except NameError:
    unicode = str

try:
    from urllib.request import urlretrieve
except BaseException:
    from urllib import urlretrieve

try:
    xrange
except NameError:
    xrange = range

# Import AllenNLP
sys.path.append(os.path.dirname(__file__) + '/third_party/')
sys.path.append(os.path.dirname(__file__) + '/third_party_mock/')
from pymagnitude.third_party.allennlp.commands.elmo import ElmoEmbedder

# Import SQLite
try:
    sys.path.append(os.path.dirname(__file__) + '/third_party/')
    sys.path.append(os.path.dirname(__file__) + '/third_party/internal/')
    from pymagnitude.third_party.internal.pysqlite2 import dbapi2 as sqlite3
    db = sqlite3.connect(':memory:')
    db.close()
    _SQLITE_LIB = 'internal'
except Exception as e:
    import sqlite3
    _SQLITE_LIB = 'system'

DEFAULT_LRU_CACHE_SIZE = 1000


def _sqlite_try_max_variable_number(num):
    """ Tests whether SQLite can handle num variables """
    db = sqlite3.connect(':memory:')
    try:
        db.cursor().execute(
            "SELECT 1 IN (" + ",".join(["?"] * num) + ")",
            ([0] * num)
        ).fetchall()
        return num
    except BaseException:
        return -1
    finally:
        db.close()


class Magnitude(object):

    SQLITE_LIB = _SQLITE_LIB
    NGRAM_BEG = 1
    NGRAM_END = DEFAULT_NGRAM_END
    BOW = BOW
    EOW = EOW
    RARE_CHAR = u"\uF002".encode('utf-8')
    FTS_SPECIAL = set('*^')
    MMAP_THREAD_LOCK = {}
    OOV_RNG_LOCK = threading.Lock()
    SQLITE_MAX_VARIABLE_NUMBER = max(max((_sqlite_try_max_variable_number(n)
                                          for n in [99, 999, 9999, 99999])), 1)
    MAX_KEY_LENGTH_FOR_OOV_SIM = 1000
    ENGLISH_PREFIXES = ['counter', 'electro', 'circum', 'contra', 'contro',
                        'crypto', 'deuter', 'franco', 'hetero', 'megalo',
                        'preter', 'pseudo', 'after', 'under', 'amphi',
                        'anglo', 'astro', 'extra', 'hydro', 'hyper', 'infra',
                        'inter', 'intra', 'micro', 'multi', 'ortho', 'paleo',
                        'photo', 'proto', 'quasi', 'retro', 'socio', 'super',
                        'supra', 'trans', 'ultra', 'anti', 'back', 'down',
                        'fore', 'hind', 'midi', 'mini', 'over', 'post',
                        'self', 'step', 'with', 'afro', 'ambi', 'ante',
                        'anti', 'arch', 'auto', 'cryo', 'demi', 'demo',
                        'euro', 'gyro', 'hemi', 'homo', 'hypo', 'ideo',
                        'idio', 'indo', 'macr', 'maxi', 'mega', 'meta',
                        'mono', 'mult', 'omni', 'para', 'peri', 'pleo',
                        'poly', 'post', 'pros', 'pyro', 'semi', 'tele',
                        'vice', 'dis', 'dis', 'mid', 'mis', 'off', 'out',
                        'pre', 'pro', 'twi', 'ana', 'apo', 'bio', 'cis',
                        'con', 'com', 'col', 'cor', 'dia', 'dis', 'dif',
                        'duo', 'eco', 'epi', 'geo', 'im ', 'iso', 'mal',
                        'mon', 'neo', 'non', 'pan', 'ped', 'per', 'pod',
                        'pre', 'pro', 'pro', 'sub', 'sup', 'sur', 'syn',
                        'syl', 'sym', 'tri', 'uni', 'be', 'by', 'co', 'de',
                        'en', 'em', 'ex', 'on', 're', 'un', 'un', 'up', 'an',
                        'an', 'ap', 'bi', 'co', 'de', 'di', 'di', 'du', 'en',
                        'el', 'em', 'ep', 'ex', 'in', 'in', 'il', 'ir', 'sy',
                        'a', 'a', 'a']
    ENGLISH_PREFIXES = sorted(
        chain.from_iterable([(p + '-', p) for p in ENGLISH_PREFIXES]),
        key=lambda x: len(x), reverse=True)
    ENGLISH_SUFFIXES = ['ification', 'ologist', 'ology', 'ology', 'able',
                        'ible', 'hood', 'ness', 'less', 'ment', 'tion',
                        'logy', 'like', 'ise', 'ize', 'ful', 'ess', 'ism',
                        'ist', 'ish', 'ity', 'ant', 'oid', 'ory', 'ing', 'fy',
                        'ly', 'al']
    ENGLISH_SUFFIXES = sorted(
        chain.from_iterable([('-' + s, s) for s in ENGLISH_SUFFIXES]),
        key=lambda x: len(x), reverse=True)

    def __new__(cls, *args, **kwargs):
        """ Returns a concatenated magnitude object, if Magnitude parameters """
        if len(args) > 0 and isinstance(args[0], Magnitude):
            obj = object.__new__(ConcatenatedMagnitude, *args, **kwargs)
            obj.__init__(*args, **kwargs)
        else:
            obj = object.__new__(cls)
        return obj

    """A Magnitude class that interfaces with the underlying SQLite
    data store to provide efficient access.

    Attributes:
        path: the file path to the magnitude file
        lazy_loading: -1 = pre-load into memory, 0 = lazy loads with unbounded
                      in-memory cache, >0 lazy loads with an LRU cache of that
                      size
        blocking: Even when lazy_loading is -1, the constructor will not block
                  it will instead pre-load into memory in a background thread,
                  if blocking is set to True, it will block until everything
                  is pre-loaded into memory
        use_numpy: Returns a NumPy array if True or a list if False
        case_insensitive: Searches for keys with case-insensitive search
        pad_to_length: Pads to a certain length if examples are shorter than
                       that length or truncates if longer than that length.
        truncate_left: if something needs to be truncated to the padding,
                       truncate off the left side
        pad_left: Pads to the left.
        placeholders: Extra empty dimensions to add to the vectors.
        ngram_oov: Use character n-grams for generating out-of-vocabulary
                   vectors.
        supress_warnings: Supress warnings generated
        batch_size: Controls the maximum vector size used in memory directly
        eager: Start loading non-critical resources in the background in
               anticipation they will be used.
        language: A ISO 639-1 Language Code (default: English 'en')
        dtype: The dtype to use when use_numpy is True.
        _number_of_values: When the path is set to None and Magnitude is being
                          used to solely featurize keys directly into vectors,
                          _number_of_values should be set to the
                          approximate upper-bound of the number of keys
                          that will be looked up with query(). If you don't know
                          the exact number, be conservative and pick a large
                          number, while keeping in mind the bigger
                          _number_of_values is, the more memory it will consume.
        _namespace: an optional namespace that will be prepended to each query
                   if provided
    """

    def __init__(self, path, lazy_loading=0, blocking=False,
                 use_numpy=True, case_insensitive=False,
                 pad_to_length=None, truncate_left=False,
                 pad_left=False, placeholders=0, ngram_oov=True,
                 supress_warnings=False, batch_size=3000000,
                 eager=True, language='en', dtype=np.float32,
                 _namespace=None, _number_of_values=1000000):
        """Initializes a new Magnitude object."""
        self.sqlite_lib = Magnitude.SQLITE_LIB
        self.closed = False
        self.uid = str(uuid.uuid4()).replace("-", "")

        self.fd = None
        if path is None:
            self.memory_db = True
            self.path = ":memory:"
        else:
            self.memory_db = False
            self.path = os.path.expanduser(path)
        self._all_conns = []
        self.lazy_loading = lazy_loading
        self.use_numpy = use_numpy
        self.case_insensitive = case_insensitive
        self.pad_to_length = pad_to_length
        self.truncate_left = truncate_left
        self.pad_left = pad_left
        self.placeholders = placeholders
        self.ngram_oov = ngram_oov
        self.supress_warnings = supress_warnings
        self.batch_size = batch_size
        self.eager = eager
        self.language = language and language.lower()
        self.dtype = dtype
        self._namespace = _namespace
        self._number_of_values = _number_of_values

        # Define conns and cursors store
        self._conns = {}
        self._cursors = {}
        self._threads = []

        # Convert the input file if not .magnitude
        if self.path.endswith('.bin') or \
                self.path.endswith('.txt') or \
                self.path.endswith('.vec') or \
                self.path.endswith('.hdf5'):
            if not supress_warnings:
                sys.stdout.write(
                    """WARNING: You are attempting to directly use a `.bin`,
                    `.txt`, `.vec`, or `.hdf5` file with Magnitude. The file is being
                    converted to the `.magnitude` format (which is slow) so
                    that it can be used with this library. This will happen on
                    every run / re-boot of your computer. If you want to make
                    this faster pre-convert your vector model to the
                    `.magnitude` format with the built-in command utility:

                    `python -m pymagnitude.converter -i input_file -o output_file`

                    Refer to the README for more information.

                    You can pass `supress_warnings=True` to the constructor to
                    hide this message.""")  # noqa
                sys.stdout.flush()
            from pymagnitude.converter_shared import convert as convert_vector_file  # noqa
            self.path = convert_vector_file(self.path)

        # If the path doesn't exist locally, try a remote download
        if not os.path.isfile(self.path) and not self.memory_db:
            self.path = MagnitudeUtils.download_model(self.path, _local=True)

        # Open a read-only file descriptor against the file
        if not self.memory_db:
            self.fd = os.open(self.path, os.O_RDONLY)

        # Get metadata about the vectors
        self.length = self._db().execute(
            "SELECT COUNT(key) FROM magnitude") \
            .fetchall()[0][0]
        version_query = self._db().execute(
            "SELECT value FROM magnitude_format WHERE key='version'") \
            .fetchall()
        self.version = version_query[0][0] if len(version_query) > 0 else 1
        elmo_query = self._db().execute(
            "SELECT value FROM magnitude_format WHERE key='elmo'") \
            .fetchall()
        self.elmo = len(elmo_query) > 0 and elmo_query[0][0]
        if CONVERTER_VERSION < self.version:
            raise RuntimeError(
                """The `.magnitude` file you are using was built with a
                newer version of Magnitude than your version of Magnitude.
                Please update the Magnitude library as it is incompatible
                with this particular `.magnitude` file.""")  # noqa
        self.original_length = self._db().execute(
            "SELECT value FROM magnitude_format WHERE key='size'") \
            .fetchall()[0][0]
        self.emb_dim = self._db().execute(
            "SELECT value FROM magnitude_format WHERE key='dim'") \
            .fetchall()[0][0]
        self.precision = self._db().execute(
            "SELECT value FROM magnitude_format WHERE key='precision'") \
            .fetchall()[0][0]
        subword_query = self._db().execute(
            "SELECT value FROM magnitude_format WHERE key='subword'") \
            .fetchall()
        self.subword = len(subword_query) > 0 and subword_query[0][0]
        if self.subword:
            self.subword_start = self._db().execute(
                "SELECT value FROM magnitude_format WHERE key='subword_start'")\
                .fetchall()[0][0]
            self.subword_end = self._db().execute(
                "SELECT value FROM magnitude_format WHERE key='subword_end'") \
                .fetchall()[0][0]
        approx_query = self._db().execute(
            "SELECT value FROM magnitude_format WHERE key='approx'") \
            .fetchall()
        self.approx = len(approx_query) > 0 and approx_query[0][0]
        if self.approx:
            self.approx_trees = self._db().execute(
                "SELECT value FROM magnitude_format WHERE key='approx_trees'")\
                .fetchall()[0][0]
        self.dim = self.emb_dim + self.placeholders
        self.highest_entropy_dimensions = [row[0] for row in self._db().execute(
            "SELECT value FROM magnitude_format WHERE key='entropy'")
            .fetchall()]
        duplicate_keys_query = self._db().execute("""
            SELECT MAX(key_count)
            FROM (
                SELECT COUNT(key)
                AS key_count
                FROM magnitude
                GROUP BY key
            );
        """).fetchall()
        self.max_duplicate_keys = (
            duplicate_keys_query[0][0] if duplicate_keys_query[0][0] is not None else 1)  # noqa

        # Iterate to pre-load
        def _preload_memory():
            if not self.eager:  # So that it doesn't loop over the vectors twice
                for key, vector in self._iter(put_cache=True):
                    pass

        # Start creating mmap in background
        self.setup_for_mmap = False
        self._all_vectors = None
        self._approx_index = None
        self._elmo_embedder = None
        if self.eager:
            mmap_thread = threading.Thread(target=self.get_vectors_mmap)
            self._threads.append(mmap_thread)
            mmap_thread.daemon = True
            mmap_thread.start()
            if self.approx:
                approx_mmap_thread = threading.Thread(
                    target=self.get_approx_index)
                self._threads.append(approx_mmap_thread)
                approx_mmap_thread.daemon = True
                approx_mmap_thread.start()
            if self.elmo:
                elmo_thread = threading.Thread(
                    target=self.get_elmo_embedder)
                self._threads.append(elmo_thread)
                elmo_thread.daemon = True
                elmo_thread.start()

        # Create cached methods
        if self.lazy_loading <= 0:
            @lru_cache(None)
            def _vector_for_key_cached(*args, **kwargs):
                return self._vector_for_key(*args, **kwargs)

            @lru_cache(None)
            def _out_of_vocab_vector_cached(*args, **kwargs):
                return self._out_of_vocab_vector(*args, **kwargs)

            @lru_cache(None)
            def _key_for_index_cached(*args, **kwargs):
                return self._key_for_index(*args, **kwargs)
            self._vector_for_key_cached = _vector_for_key_cached
            self._out_of_vocab_vector_cached = _out_of_vocab_vector_cached
            self._key_for_index_cached = _key_for_index_cached
            if self.lazy_loading == -1:
                if blocking:
                    _preload_memory()
                else:
                    preload_thread = threading.Thread(target=_preload_memory)
                    self._threads.append(preload_thread)
                    preload_thread.daemon = True
                    preload_thread.start()
        elif self.lazy_loading > 0:
            @lru_cache(self.lazy_loading)
            def _vector_for_key_cached(*args, **kwargs):
                return self._vector_for_key(*args, **kwargs)

            @lru_cache(self.lazy_loading)
            def _out_of_vocab_vector_cached(*args, **kwargs):
                return self._out_of_vocab_vector(*args, **kwargs)

            @lru_cache(self.lazy_loading)
            def _key_for_index_cached(*args, **kwargs):
                return self._key_for_index(*args, **kwargs)
            self._vector_for_key_cached = _vector_for_key_cached
            self._out_of_vocab_vector_cached = _out_of_vocab_vector_cached
            self._key_for_index_cached = _key_for_index_cached

        if self.eager and blocking:
            self.get_vectors_mmap()  # Wait for mmap to be available
            if self.approx:
                self.get_approx_index()  # Wait for approx mmap to be available
            if self.elmo:
                self.get_elmo_embedder()  # Wait for approx mmap to be available

    def _setup_for_mmap(self):
        # Setup variables for get_vectors_mmap()
        self._all_vectors = None
        self._approx_index = None
        self._elmo_embedder = None
        if not self.memory_db:
            self.db_hash = fast_md5_file(self.path)
        else:
            self.db_hash = self.uid
        self.md5 = hashlib.md5(",".join(
            [self.path, self.db_hash, str(self.length),
             str(self.dim), str(self.precision), str(self.case_insensitive)
             ]).encode('utf-8')).hexdigest()
        self.path_to_mmap = os.path.join(tempfile.gettempdir(),
                                         self.md5 + '.magmmap')
        self.path_to_approx_mmap = os.path.join(tempfile.gettempdir(),
                                                self.md5 + '.approx.magmmap')
        self.path_to_elmo_w_mmap = os.path.join(tempfile.gettempdir(),
                                                self.md5 + '.elmo.hdf5.magmmap')
        self.path_to_elmo_o_mmap = os.path.join(tempfile.gettempdir(),
                                                self.md5 + '.elmo.json.magmmap')
        if self.path_to_mmap not in Magnitude.MMAP_THREAD_LOCK:
            Magnitude.MMAP_THREAD_LOCK[self.path_to_mmap] = threading.Lock()
        if self.path_to_approx_mmap not in Magnitude.MMAP_THREAD_LOCK:
            Magnitude.MMAP_THREAD_LOCK[self.path_to_approx_mmap] = \
                threading.Lock()
        if self.path_to_elmo_w_mmap not in Magnitude.MMAP_THREAD_LOCK:
            Magnitude.MMAP_THREAD_LOCK[self.path_to_elmo_w_mmap] = \
                threading.Lock()
        if self.path_to_elmo_o_mmap not in Magnitude.MMAP_THREAD_LOCK:
            Magnitude.MMAP_THREAD_LOCK[self.path_to_elmo_o_mmap] = \
                threading.Lock()
        self.MMAP_THREAD_LOCK = Magnitude.MMAP_THREAD_LOCK[self.path_to_mmap]
        self.MMAP_PROCESS_LOCK = InterProcessLock(self.path_to_mmap + '.lock')
        self.APPROX_MMAP_THREAD_LOCK = \
            Magnitude.MMAP_THREAD_LOCK[self.path_to_approx_mmap]
        self.APPROX_MMAP_PROCESS_LOCK = \
            InterProcessLock(self.path_to_approx_mmap + '.lock')
        self.ELMO_W_MMAP_THREAD_LOCK = \
            Magnitude.MMAP_THREAD_LOCK[self.path_to_elmo_w_mmap]
        self.ELMO_W_MMAP_PROCESS_LOCK = \
            InterProcessLock(self.path_to_elmo_w_mmap + '.lock')
        self.ELMO_O_MMAP_THREAD_LOCK = \
            Magnitude.MMAP_THREAD_LOCK[self.path_to_elmo_o_mmap]
        self.ELMO_O_MMAP_PROCESS_LOCK = \
            InterProcessLock(self.path_to_elmo_o_mmap + '.lock')
        self.setup_for_mmap = True

    def _db(self, force_new=False):
        """Returns a cursor to the database. Each thread gets its
        own cursor.
        """
        identifier = threading.current_thread().ident
        conn_exists = identifier in self._cursors
        if not conn_exists or force_new:
            if self.fd:
                if os.name == 'nt':
                    conn = sqlite3.connect(self.path,
                                           check_same_thread=False)
                else:
                    conn = sqlite3.connect('/dev/fd/%d' % self.fd,
                                           check_same_thread=False)
            else:
                conn = sqlite3.connect(self.path, check_same_thread=False)
                self._create_empty_db(conn.cursor())
            self._all_conns.append(conn)
        if not conn_exists:
            self._conns[identifier] = conn
            self._cursors[identifier] = conn.cursor()
        elif force_new:
            return conn.cursor()
        return self._cursors[identifier]

    def _create_empty_db(self, db):
        # Calculates the number of dimensions needed to prevent hashing from
        # creating a collision error of a certain value for the number of
        # expected feature values being hashed
        collision_error_allowed = .001
        number_of_dims = max(math.ceil(math.log(
            ((self._number_of_values ** 2) / (-2 * math.log(-collision_error_allowed + 1))), 100)), 2)  # noqa

        db.execute("DROP TABLE IF EXISTS `magnitude`;")
        db.execute("""
            CREATE TABLE `magnitude` (
                key TEXT COLLATE NOCASE
            );
        """)
        db.execute("""
            CREATE TABLE `magnitude_format` (
                key TEXT COLLATE NOCASE,
                value INTEGER
            );
        """)
        insert_format_query = """
            INSERT INTO `magnitude_format`(
                key,
                value
            )
            VALUES (
                ?, ?
            );
        """
        db.execute(insert_format_query, ('size', 0))
        db.execute(insert_format_query, ('dim', number_of_dims))
        db.execute(insert_format_query, ('precision', 0))

    def _padding_vector(self):
        """Generates a padding vector."""
        if self.use_numpy:
            return np.zeros((self.dim,), dtype=self.dtype)
        else:
            return [0.0] * self.dim

    def _key_t(self, key):
        """Transforms a key to lower case depending on case
        sensitivity.
        """
        if self.case_insensitive and (isinstance(key, str) or
                                      isinstance(key, unicode)):
            return key.lower()
        return key

    def _string_dist(self, a, b):
        length = max(len(a), len(b))
        return length - difflib.SequenceMatcher(None, a, b).ratio() * length

    def _key_shrunk_2(self, key):
        """Shrinks more than two characters to two characters
        """
        return re.sub(r"([^<])\1{2,}", r"\1\1", key)

    def _key_shrunk_1(self, key):
        """Shrinks more than one character to a single character
        """
        return re.sub(r"([^<])\1+", r"\1", key)

    def _oov_key_t(self, key):
        """Transforms a key for out-of-vocabulary lookup.
        """
        is_str = isinstance(key, str) or isinstance(key, unicode)
        if is_str:
            key = Magnitude.BOW + self._key_t(key) + Magnitude.EOW
            return is_str, self._key_shrunk_2(key)
        return is_str, key

    def _oov_english_stem_english_ixes(self, key):
        """Strips away common English prefixes and suffixes."""
        key_lower = key.lower()
        start_idx = 0
        end_idx = 0
        for p in Magnitude.ENGLISH_PREFIXES:
            if key_lower[:len(p)] == p:
                start_idx = len(p)
                break
        for s in Magnitude.ENGLISH_SUFFIXES:
            if key_lower[-len(s):] == s:
                end_idx = len(s)
                break
        start_idx = start_idx if max(start_idx, end_idx) == start_idx else 0
        end_idx = end_idx if max(start_idx, end_idx) == end_idx else 0
        stripped_key = key[start_idx:len(key) - end_idx]
        if len(stripped_key) < 4:
            return key
        elif stripped_key != key:
            return self._oov_english_stem_english_ixes(stripped_key)
        else:
            return stripped_key

    def _oov_stem(self, key):
        """Strips away common prefixes and suffixes."""
        if self.language == 'en':
            return self._oov_english_stem_english_ixes(key)
        return key

    def _db_query_similar_keys_vector(self, key, orig_key, topn=3):
        """Finds similar keys in the database and gets the mean vector."""
        def _sql_escape_single(s):
            return s.replace("'", "''")

        def _sql_escape_fts(s):
            return ''.join("\\" + c if c in Magnitude.FTS_SPECIAL
                           else c for c in s).replace('"', '""')

        exact_search_query = """
            SELECT *
            FROM `magnitude`
            WHERE key = ?
            ORDER BY key = ? COLLATE NOCASE DESC
            LIMIT ?;
        """

        if self.subword and len(key) < Magnitude.MAX_KEY_LENGTH_FOR_OOV_SIM:
            current_subword_start = self.subword_end
            BOW_length = len(Magnitude.BOW)  # noqa: N806
            EOW_length = len(Magnitude.EOW)  # noqa: N806
            BOWEOW_length = BOW_length + EOW_length  # noqa: N806
            true_key_len = len(key) - BOWEOW_length
            key_shrunk_stemmed = self._oov_stem(self._key_shrunk_1(orig_key))
            key_shrunk = self._key_shrunk_1(orig_key)
            key_stemmed = self._oov_stem(orig_key)
            beginning_and_end_clause = ""
            exact_matches = []
            if true_key_len <= 6:
                beginning_and_end_clause = """
                    magnitude.key LIKE '{0}%'
                        AND LENGTH(magnitude.key) <= {2} DESC,
                    magnitude.key LIKE '%{1}'
                        AND LENGTH(magnitude.key) <= {2} DESC,"""
                beginning_and_end_clause = beginning_and_end_clause.format(
                    _sql_escape_single(key[BOW_length:BOW_length + 1]),
                    _sql_escape_single(key[-EOW_length - 1:-EOW_length]),
                    str(true_key_len))
            if key != orig_key:
                exact_matches.append((key_shrunk, self._key_shrunk_2(orig_key)))
            if key_stemmed != orig_key:
                exact_matches.append((key_stemmed,))
            if key_shrunk_stemmed != orig_key:
                exact_matches.append((key_shrunk_stemmed,))
            if len(exact_matches) > 0:
                for exact_match in exact_matches:
                    results = []
                    split_results = []
                    limits = np.array_split(list(range(topn)), len(exact_match))
                    for i, e in enumerate(exact_match):
                        limit = len(limits[i])
                        split_results.extend(self._db().execute(
                            exact_search_query, (e, e, limit)).fetchall())
                        results.extend(self._db().execute(
                            exact_search_query, (e, e, topn)).fetchall())
                    if len(split_results) >= topn:
                        results = split_results
                    if len(results) > 0:
                        break
            else:
                results = []
            if len(results) == 0:
                search_query = """
                    SELECT magnitude.*
                    FROM magnitude_subword, magnitude
                    WHERE char_ngrams MATCH ?
                    AND magnitude.rowid = magnitude_subword.rowid
                    ORDER BY
                        (
                            (
                                LENGTH(offsets(magnitude_subword)) -
                                LENGTH(
                                    REPLACE(offsets(magnitude_subword), ' ', '')
                                )
                            )
                            +
                            1
                        ) DESC,
                        """ + beginning_and_end_clause + """
                        LENGTH(magnitude.key) ASC
                    LIMIT ?;
                """  # noqa
                while (len(results) < topn and
                        current_subword_start >= self.subword_start):
                    ngrams = list(char_ngrams(
                        key, current_subword_start, current_subword_start))
                    ngram_limit_map = {
                        6: 4,
                        5: 8,
                        4: 12,
                    }
                    while current_subword_start in ngram_limit_map and len(
                            ngrams) > ngram_limit_map[current_subword_start]:
                        # Reduce the search parameter space by sampling every
                        # other ngram
                        ngrams = ngrams[:-1][::2] + ngrams[-1:]
                    params = (' OR '.join('"{0}"'.format(_sql_escape_fts(n))
                                          for n in ngrams), topn)
                    results = self._db().execute(search_query,
                                                 params).fetchall()
                    small_typo = len(results) > 0 and self._string_dist(
                        results[0][0].lower(), orig_key.lower()) <= 4
                    if key_shrunk_stemmed != orig_key and key_shrunk_stemmed != key_shrunk and not small_typo:  # noqa
                        ngrams = list(
                            char_ngrams(
                                self._oov_key_t(key_shrunk_stemmed)[1],
                                current_subword_start,
                                self.subword_end))
                        params = (' OR '.join('"{0}"'.format(_sql_escape_fts(n))
                                              for n in ngrams), topn)
                        results = self._db().execute(search_query,
                                                     params).fetchall()
                    current_subword_start -= 1
        else:
            # As a backup do a search with 'NOCASE'
            results = self._db().execute(exact_search_query,
                                         (orig_key, orig_key, topn)).fetchall()
        final_results = []
        for result in results:
            result_key, vec = self._db_full_result_to_vec(result)
            final_results.append(vec)
        if len(final_results) > 0:
            mean_vector = np.mean(final_results, axis=0)
            return mean_vector / np.linalg.norm(mean_vector)
        else:
            return self._padding_vector()

    def _seed(self, val):
        """Returns a unique seed for val and the (optional) namespace."""
        if self._namespace:
            return xxhash.xxh32(
                self._namespace.encode('utf-8') +
                Magnitude.RARE_CHAR +
                val.encode('utf-8')).intdigest()
        else:
            return xxhash.xxh32(val.encode('utf-8')).intdigest()

    def _out_of_vocab_vector(self, key):
        """Generates a random vector based on the hash of the key."""
        orig_key = key
        is_str, key = self._oov_key_t(key)
        if self.elmo and is_str:
            elmo_oov = np.concatenate(self.get_elmo_embedder().embed_batch(
                [[key]])[0], axis=1).flatten()
            elmo_oov = elmo_oov / np.linalg.norm(elmo_oov)
            if self.use_numpy:
                return elmo_oov
            else:
                return elmo_oov.tolist()

        if not is_str:
            seed = self._seed(type(key).__name__)
            Magnitude.OOV_RNG_LOCK.acquire()
            np.random.seed(seed=seed)
            random_vector = np.random.uniform(-1, 1, (self.emb_dim,))
            Magnitude.OOV_RNG_LOCK.release()
            random_vector[-1] = self.dtype(key) / np.finfo(self.dtype).max
        elif not self.ngram_oov or len(key) < Magnitude.NGRAM_BEG:
            seed = self._seed(key)
            Magnitude.OOV_RNG_LOCK.acquire()
            np.random.seed(seed=seed)
            random_vector = np.random.uniform(-1, 1, (self.emb_dim,))
            Magnitude.OOV_RNG_LOCK.release()
        else:
            ngrams = char_ngrams(key, Magnitude.NGRAM_BEG,
                                 Magnitude.NGRAM_END)
            random_vectors = []
            for i, ngram in enumerate(ngrams):
                seed = self._seed(ngram)
                Magnitude.OOV_RNG_LOCK.acquire()
                np.random.seed(seed=seed)
                random_vectors.append(
                    np.random.uniform(-1, 1, (self.emb_dim,)))
                Magnitude.OOV_RNG_LOCK.release()
            random_vector = np.mean(random_vectors, axis=0)

        np.random.seed()
        if self.placeholders > 0:
            random_vector = np.pad(random_vector, [(0, self.placeholders)],
                                   mode='constant', constant_values=0.0)
        if is_str:
            random_vector = random_vector / np.linalg.norm(random_vector)
            final_vector = (
                random_vector *
                0.3 +
                self._db_query_similar_keys_vector(
                    key,
                    orig_key) *
                0.7)
            final_vector = final_vector / np.linalg.norm(final_vector)
        else:
            final_vector = random_vector
        if self.use_numpy:
            return final_vector
        else:
            return final_vector.tolist()

    def _db_batch_generator(self, params):
        """ Generates batches of paramaters that respect
        SQLite's MAX_VARIABLE_NUMBER """
        if len(params) <= Magnitude.SQLITE_MAX_VARIABLE_NUMBER:
            yield params
        else:
            it = iter(params)
            for batch in \
                    iter(lambda: tuple(
                        islice(it, Magnitude.SQLITE_MAX_VARIABLE_NUMBER)
                    ), ()):
                yield batch

    def _db_result_to_vec(self, result):
        """Converts a database result to a vector."""
        if self.use_numpy:
            vec = np.zeros((self.dim,), dtype=self.dtype)
            vec[0:self.emb_dim] = result
            vec = vec / float(10**self.precision)
            return vec
        else:
            return [v / float(10**self.precision) for v in result] + \
                [0.0] * self.placeholders

    def _db_full_result_to_vec(self, result, put_cache=True):
        """Converts a full database result to a vector."""
        result_key = result[0]
        if self._query_is_cached(result_key):
            return (result_key, self.query(result_key))
        else:
            vec = self._db_result_to_vec(result[1:])
            if put_cache:
                self._vector_for_key_cached._cache.put((result_key,), vec)
            return (result_key, vec)

    def _vector_for_key(self, key):
        """Queries the database for a single key."""
        result = self._db().execute(
            """
                SELECT *
                FROM `magnitude`
                WHERE key = ?
                ORDER BY key = ? COLLATE BINARY DESC
                LIMIT 1;""",
            (key, key)).fetchone()
        if result is None or self._key_t(result[0]) != self._key_t(key):
            return None
        else:
            return self._db_result_to_vec(result[1:])

    def _vectors_for_keys(self, keys):
        """Queries the database for multiple keys."""
        if self.elmo:
            keys = [self._key_t(key) for key in keys]
            elmo_ctx = np.concatenate(self.get_elmo_embedder().embed_batch(
                [keys])[0], axis=1)
            elmo_ctx = norm_matrix(elmo_ctx)
            if self.use_numpy:
                return elmo_ctx
            else:
                return elmo_ctx.tolist()

        unseen_keys = tuple(key for key in keys
                            if not self._query_is_cached(key))
        unseen_keys_map = {}
        if len(unseen_keys) > 0:
            unseen_keys_map = {self._key_t(k): i for i, k in
                               enumerate(unseen_keys)}
            unseen_vectors = [None] * len(unseen_keys)
            seen_keys = set()
            for unseen_keys_batch in self._db_batch_generator(unseen_keys):
                results = self._db().execute(
                    """
                        SELECT *
                        FROM `magnitude`
                        WHERE key
                        IN (""" + ' ,'.join(['?'] * len(unseen_keys_batch)) +
                    """);
                    """,
                    unseen_keys_batch)
                for result in results:
                    result_key, vec = self._db_full_result_to_vec(result)
                    result_key_t = self._key_t(result_key)
                    if result_key_t in unseen_keys_map:
                        i = unseen_keys_map[result_key_t]
                        if (
                            (result_key_t not in seen_keys or
                             result_key == unseen_keys[i]) and

                            (
                                self.case_insensitive or
                                result_key == unseen_keys[i])
                        ):
                            seen_keys.add(result_key_t)
                            unseen_vectors[i] = vec
            for i in range(len(unseen_vectors)):
                self._vector_for_key_cached._cache.put((unseen_keys[i],),
                                                       unseen_vectors[i])
                if unseen_vectors[i] is None:
                    unseen_vectors[i] = \
                        self._out_of_vocab_vector_cached(unseen_keys[i])
        vectors = [self.query(key) if key not in unseen_keys_map else
                   unseen_vectors[unseen_keys_map[self._key_t(key)]]
                   for key in keys]
        return vectors

    def _key_for_index(self, index, return_vector=True):
        """Queries the database the key at a single index."""
        columns = "key"
        if return_vector:
            columns = "*"
        result = self._db().execute(
            """
                SELECT """ + columns + """
                FROM `magnitude`
                WHERE rowid = ?
                LIMIT 1;
            """,
            (int(index + 1),)).fetchone()
        if result is None:
            raise IndexError("The index %d is out-of-range" % index)
        else:
            if return_vector:
                return self._db_full_result_to_vec(result)
            else:
                return result[0]

    def _keys_for_indices(self, indices, return_vector=True):
        """Queries the database for the keys of multiple indices."""
        unseen_indices = tuple(int(index + 1) for index in indices
                               if self._key_for_index_cached._cache.get(((index,),  # noqa
                                                                         frozenset([('return_vector', return_vector)]))) is None)  # noqa
        unseen_indices_map = {}
        if len(unseen_indices) > 0:
            columns = "key"
            if return_vector:
                columns = "*"
            unseen_indices_map = {(index - 1): i for i, index in
                                  enumerate(unseen_indices)}
            unseen_keys = [None] * len(unseen_indices)
            for unseen_indices_batch in \
                    self._db_batch_generator(unseen_indices):
                results = self._db().execute(
                    """
                        SELECT rowid, """ + columns + """
                        FROM `magnitude`
                        WHERE rowid IN (""" +
                    ' ,'.join(['?'] * len(unseen_indices_batch)) +
                    """);""",
                    unseen_indices_batch)
                for result in results:
                    i = unseen_indices_map[result[0] - 1]
                    result_key = result[1]
                    if return_vector:
                        unseen_keys[i] = self._db_full_result_to_vec(
                            result[1:])
                    else:
                        unseen_keys[i] = result_key
                    self._key_for_index_cached._cache.put(
                        (
                            (unseen_indices[i] - 1,),
                            frozenset([('return_vector', return_vector)])
                        ),
                        unseen_keys[i]
                    )
            for i in range(len(unseen_keys)):
                if unseen_keys[i] is None:
                    raise IndexError("The index %d is out-of-range" %
                                     unseen_indices[i] - 1)
        keys = [self.index(index, return_vector=return_vector)
                if index not in unseen_indices_map else
                unseen_keys[unseen_indices_map[index]] for index in indices]
        return keys

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def query(self, q, pad_to_length=None,
              pad_left=None, truncate_left=None):
        """Handles a query of keys which could be a single key, a
        1-D list of keys, or a 2-D list of keys.
        """
        pad_to_length = pad_to_length or self.pad_to_length
        pad_left = pad_left or self.pad_left
        truncate_left = truncate_left or self.truncate_left

        if not isinstance(q, list):  # Single key
            vec = self._vector_for_key_cached(q)
            if vec is None:
                return self._out_of_vocab_vector_cached(q)
            else:
                return vec
        elif isinstance(q, list) \
                and (len(q) == 0 or not isinstance(q[0], list)):  # 1D list
            pad_to_length = pad_to_length if pad_to_length else len(q)
            padding_length = max(pad_to_length - len(q), 0)
            keys_length = pad_to_length - padding_length
            vectors = self._vectors_for_keys(q)
            if truncate_left:
                vectors = vectors[-keys_length:]
            else:
                vectors = vectors[0:keys_length]
            if self.use_numpy:
                tensor = np.zeros((pad_to_length, self.dim), dtype=self.dtype)
            else:
                tensor = [self._padding_vector() for i in range(pad_to_length)]
            if pad_left:
                tensor[-keys_length:] = vectors
            else:
                tensor[0:keys_length] = vectors
            return tensor
        elif isinstance(q, list):  # 2D List
            max_q = max([len(subquery) for subquery in q])
            pad_to_length = pad_to_length if pad_to_length else max_q
            if self.use_numpy:
                tensor = np.zeros((len(q), pad_to_length, self.dim),
                                  dtype=self.dtype)
            else:
                tensor = [[self._padding_vector() for i in range(pad_to_length)]
                          for j in range(len(q))]
            for row, sq in enumerate(q):
                padding_length = max(pad_to_length - len(sq), 0)
                keys_length = pad_to_length - padding_length
                vectors = self._vectors_for_keys(sq)
                if truncate_left:
                    vectors = vectors[-keys_length:]
                else:
                    vectors = vectors[0:keys_length]
                if pad_left:
                    if self.use_numpy:
                        tensor[row, -keys_length:] = vectors
                    else:
                        tensor[row][-keys_length:] = vectors
                else:
                    if self.use_numpy:
                        tensor[row, 0:keys_length] = vectors
                    else:
                        tensor[row][0:keys_length] = vectors
            return tensor

    def index(self, q, return_vector=True):
        """Gets a key for an index or multiple indices."""
        if isinstance(q, list) or isinstance(q, tuple):
            return self._keys_for_indices(q, return_vector=return_vector)
        else:
            return self._key_for_index_cached(q, return_vector=return_vector)

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def _query_numpy(self, key):
        """Returns the query for a key, forcibly converting the
        resulting vector to a numpy array.
        """
        key_is_ndarray = isinstance(key, np.ndarray)
        key_is_list = isinstance(key, list)
        key_len_ge_0 = key_is_list and len(key) > 0
        key_0_is_number = key_len_ge_0 and isinstance(key[0], Number)
        key_0_is_ndarray = key_len_ge_0 and isinstance(key[0], np.ndarray)
        key_0_is_list = key_len_ge_0 and isinstance(key[0], list)
        key_0_len_ge_0 = key_0_is_list and len(key[0]) > 0
        key_0_0_is_number = (key_0_is_list and key_0_len_ge_0 and
                             isinstance(key[0][0], Number))
        if (key_is_ndarray or key_0_is_number or key_0_is_ndarray or key_0_0_is_number):  # noqa
            return key
        elif not self.use_numpy:
            return np.asarray(self.query(key))
        else:
            return self.query(key)

    def _query_is_cached(self, key):
        """Checks if the query been cached by Magnitude."""
        return ((self._vector_for_key_cached._cache.get((key,)) is not None) or (  # noqa
            self._out_of_vocab_vector_cached._cache.get((key,)) is not None))

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def distance(self, key, q):
        """Calculates the distance from key to the key(s) in q."""
        a = self._query_numpy(key)
        if not isinstance(q, list):
            b = self._query_numpy(q)
            return np.linalg.norm(a - b)
        else:
            return [np.linalg.norm(a - self._query_numpy(b)) for b in q]

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def similarity(self, key, q):
        """Calculates the similarity from key to the key(s) in q."""
        a = self._query_numpy(key)
        if not isinstance(q, list):
            b = self._query_numpy(q)
            return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        else:
            bs = [self._query_numpy(b) for b in q]
            return [np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                    for b in bs]

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def most_similar_to_given(self, key, q):
        """Calculates the most similar key in q to key."""
        distances = self.distance(key, q)
        min_index, _ = min(enumerate(distances), key=operator.itemgetter(1))
        return q[min_index]

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def doesnt_match(self, q):
        """Given a set of keys, figures out which key doesn't
        match the rest.
        """
        mean_vector = np.mean(self._query_numpy([[sq] for sq in q]), axis=0)
        mean_unit_vector = mean_vector / np.linalg.norm(mean_vector)
        distances = [np.linalg.norm(mean_unit_vector - self._query_numpy(b))
                     for b in q]
        max_index, _ = max(enumerate(distances), key=operator.itemgetter(1))
        return q[max_index]

    def _db_query_similarity(
            self,
            positive,
            negative,
            min_similarity=None,
            topn=10,
            exclude_keys=set(),
            return_similarities=False,
            method='distance',
            effort=1.0):
        """Runs a database query to find vectors close to vector."""
        COSMUL = method == '3cosmul'  # noqa: N806
        APPROX = method == 'approx'  # noqa: N806
        DISTANCE = not COSMUL and not APPROX  # noqa: N806

        exclude_keys = {self._key_t(exclude_key)
                        for exclude_key in exclude_keys}

        if topn is None:
            topn = self.length

        filter_topn = self.max_duplicate_keys * (topn + len(exclude_keys))

        # Find mean unit vector
        if (DISTANCE or APPROX) and (len(negative) > 0 or len(positive) > 1):
            positive_vecs = np.sum(self._query_numpy(positive), axis=0)
            if len(negative) > 0:
                negative_vecs = -1.0 * np.sum(self._query_numpy(negative),
                                              axis=0)
            else:
                negative_vecs = np.zeros((self.dim,), dtype=self.dtype)
            mean_vector = (positive_vecs + negative_vecs) / \
                float(len(positive) + len(negative))
            mean_unit_vector = mean_vector / np.linalg.norm(mean_vector)
        elif (DISTANCE or APPROX):
            mean_unit_vector = self._query_numpy(positive[0])
        elif COSMUL:
            positive_vecs = self._query_numpy(positive)
            if len(negative) > 0:
                negative_vecs = self._query_numpy(negative)
            else:
                negative_vecs = np.zeros((0, self.dim))

        # Calculate topn closest in batches over all vectors
        if DISTANCE or COSMUL:
            filtered_indices = []
            for batch_start, _, batch in \
                    self.get_vectors_mmap_batch_generator():
                if DISTANCE:
                    similiarities = np.dot(batch, mean_unit_vector)
                elif COSMUL:
                    positive_similiarities = [
                        ((1 + np.dot(batch, vec)) / 2)
                        for vec in positive_vecs
                    ]
                    negative_similiarities = [
                        ((1 + np.dot(batch, vec)) / 2)
                        for vec in negative_vecs
                    ]
                    similiarities = (
                        np.prod(positive_similiarities, axis=0) /
                        (np.prod(negative_similiarities, axis=0) + 0.000001))

                partition_results = np.argpartition(similiarities, -1 * min(
                    filter_topn, self.batch_size, self.length))[-filter_topn:]

                for index in partition_results:
                    if (min_similarity is None or
                            similiarities[index] >= min_similarity):
                        if len(filtered_indices) < filter_topn:
                            heapq.heappush(filtered_indices, (
                                similiarities[index],
                                batch_start + index))
                        elif similiarities[index] > filtered_indices[0][0]:
                            heapq.heappushpop(filtered_indices, (
                                similiarities[index],
                                batch_start + index))

            # Get the final topn from all batches
            topn_indices = heapq.nlargest(filter_topn, filtered_indices,
                                          key=lambda x: x[0])
            topn_indices = iter(topn_indices)
        elif APPROX:
            approx_index = self.get_approx_index()
            search_k = int(effort * filter_topn * self.approx_trees)
            nns = approx_index.get_nns_by_vector(
                mean_unit_vector,
                filter_topn,
                search_k=search_k,
                include_distances=True)
            topn_indices = izip(nns[1], nns[0])
            topn_indices = imap(lambda di: (1 - di[0] ** 2 * .5, di[1]),
                                topn_indices)

        # Tee topn_indices iterator
        topn_indices_1, topn_indices_2 = tee(topn_indices)

        # Retrieve the keys of the vectors
        keys = self.index([i[1] for i in topn_indices_1],
                          return_vector=False)

        # Build the result
        results = []
        for key, similarity in izip(keys, topn_indices_2):
            key_t = self._key_t(key)
            if len(results) >= topn:
                break
            if key_t in exclude_keys:
                continue
            exclude_keys.add(key_t)
            if return_similarities:
                results.append((key, similarity[0]))
            else:
                results.append(key)
        return results

    def _handle_pos_neg_args(self, positive, negative):
        if not isinstance(
                positive,
                list) or (
                len(positive) > 0 and isinstance(
                positive[0],
                Number)):
            positive = [positive]
        if not isinstance(
                negative,
                list) or (
                len(negative) > 0 and isinstance(
                negative[0],
                Number)):
            negative = [negative]
        return positive, negative

    def _exclude_set(self, positive, negative):
        def _is_vec(elem):
            return isinstance(elem, np.ndarray) or \
                (isinstance(elem, list) and len(elem) > 0 and
                 isinstance(elem[0], Number))

        return frozenset((elem for elem in chain.from_iterable(
            [positive, negative]) if not _is_vec(elem)))

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def most_similar(self, positive, negative=[], topn=10, min_similarity=None,
                     return_similarities=True):
        """Finds the topn most similar vectors under or equal
        to max distance.
        """
        positive, negative = self._handle_pos_neg_args(positive, negative)

        return self._db_query_similarity(
            positive=positive,
            negative=negative,
            min_similarity=min_similarity,
            topn=topn,
            exclude_keys=self._exclude_set(
                positive,
                negative),
            return_similarities=return_similarities,
            method='distance')

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def most_similar_cosmul(self, positive, negative=[], topn=10,
                            min_similarity=None, return_similarities=True):
        """Finds the topn most similar vectors under or equal to max
        distance using 3CosMul:
        [Levy and Goldberg](http://www.aclweb.org/anthology/W14-1618)
        """

        positive, negative = self._handle_pos_neg_args(positive, negative)

        results = self._db_query_similarity(
            positive=positive,
            negative=negative,
            min_similarity=min_similarity,
            topn=topn,
            exclude_keys=self._exclude_set(
                positive,
                negative),
            return_similarities=return_similarities,
            method='3cosmul')
        return results

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def most_similar_approx(
            self,
            positive,
            negative=[],
            topn=10,
            min_similarity=None,
            return_similarities=True,
            effort=1.0):
        """Approximates the topn most similar vectors under or equal to max
        distance using Annoy:
        https://github.com/spotify/annoy
        """
        if not self.approx:
            raise RuntimeError("The `.magnitude` file you are using does not \
support the `most_similar_approx` function. If you are using a pre-built \
`.magnitude` file, visit Magnitude's git repository page's README and download \
the 'Heavy' model instead. If you converted this `.magnitude` file yourself \
you will need to re-convert the file passing the `-a` flag to the converter to \
build the appropriate indexes into the `.magnitude` file.")

        positive, negative = self._handle_pos_neg_args(positive, negative)

        effort = min(max(0, effort), 1.0)

        results = self._db_query_similarity(
            positive=positive,
            negative=negative,
            min_similarity=min_similarity,
            topn=topn,
            exclude_keys=self._exclude_set(
                positive,
                negative),
            return_similarities=return_similarities,
            method='approx',
            effort=effort)
        return results

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def closer_than(self, key, q, topn=None):
        """Finds all keys closer to key than q is to key."""
        epsilon = (10.0 / 10**6)
        min_similarity = self.similarity(key, q) + epsilon

        return self.most_similar(key, topn=topn, min_similarity=min_similarity,
                                 return_similarities=False)

    def get_vectors_mmap(self):
        """Gets a numpy.memmap of all vectors, blocks if it is still
        being built.
        """
        if self._all_vectors is None:
            while True:
                if not self.setup_for_mmap:
                    self._setup_for_mmap()
                try:
                    if not self.memory_db:
                        all_vectors = np.memmap(
                            self.path_to_mmap, dtype=self.dtype, mode='r',
                            shape=(self.length, self.dim))
                        self._all_vectors = all_vectors
                    else:
                        all_vectors = np.zeros((0, self.dim))
                        self._all_vectors = all_vectors
                    break
                except BaseException:
                    path_to_mmap_temp = self.path_to_mmap + '.tmp'
                    tlock = self.MMAP_THREAD_LOCK.acquire(False)
                    plock = self.MMAP_PROCESS_LOCK.acquire(0)
                    if tlock and plock:
                        values = imap(
                            lambda kv: kv[1], self._iter(
                                put_cache=self.lazy_loading == -1))
                        try:
                            with open(path_to_mmap_temp, "w+b") as mmap_file:
                                all_vectors = np.memmap(
                                    mmap_file, dtype=self.dtype, mode='w+',
                                    shape=(self.length, self.dim))
                                for i, value in enumerate(values):
                                    all_vectors[i] = value
                                all_vectors.flush()
                                del all_vectors
                            if not self.closed:
                                os.rename(path_to_mmap_temp, self.path_to_mmap)
                            else:
                                return
                        finally:
                            self.MMAP_THREAD_LOCK.release()
                            try:
                                self.MMAP_PROCESS_LOCK.release()
                            except BaseException:
                                pass
                sleep(1)  # Block before trying again
        return self._all_vectors

    def get_vectors_mmap_batch_generator(self):
        """Gets batches of get_vectors_mmap()."""
        all_vectors = self.get_vectors_mmap()

        if self.length > self.batch_size:
            for i in range(all_vectors.shape[0]):
                batch_start = i * self.batch_size
                batch_end = min(batch_start + self.batch_size,
                                all_vectors.shape[0])
                if batch_start >= all_vectors.shape[0]:
                    break
                yield (batch_start, batch_end,
                       all_vectors[batch_start:batch_end])
                if batch_end == all_vectors.shape[0]:
                    break
        else:
            yield (0, self.length, all_vectors)

    def get_approx_index_chunks(self):
        """Gets decompressed chunks of the AnnoyIndex of the vectors from
        the database."""
        try:
            db = self._db(force_new=True)
            with lz4.frame.LZ4FrameDecompressor() as decompressor:
                chunks = db.execute(
                    """
                        SELECT rowid,index_file
                        FROM `magnitude_approx`
                        WHERE trees = ?
                    """, (self.approx_trees,))
                for chunk in chunks:
                    yield decompressor.decompress(chunk[1])
                    if self.closed:
                        return
        except Exception as e:
            if self.closed:
                pass
            else:
                raise e

    def get_meta_chunks(self, meta_index):
        """Gets decompressed chunks of a meta file embedded in
        the database."""
        try:
            db = self._db(force_new=True)
            with lz4.frame.LZ4FrameDecompressor() as decompressor:
                chunks = db.execute(
                    """
                        SELECT rowid,meta_file
                        FROM `magnitude_meta_""" + str(meta_index) + """`
                    """)
                for chunk in chunks:
                    yield decompressor.decompress(chunk[1])
                    if self.closed:
                        return
        except Exception as e:
            if self.closed:
                pass
            else:
                raise e

    def get_approx_index(self):
        """Gets an AnnoyIndex of the vectors from the database."""
        chunks = self.get_approx_index_chunks()
        if self._approx_index is None:
            while True:
                if not self.setup_for_mmap:
                    self._setup_for_mmap()
                try:
                    approx_index = AnnoyIndex(self.emb_dim, metric='angular')
                    approx_index.load(self.path_to_approx_mmap)
                    self._approx_index = approx_index
                    break
                except BaseException:
                    path_to_approx_mmap_temp = self.path_to_approx_mmap \
                        + '.tmp'
                    tlock = self.APPROX_MMAP_THREAD_LOCK.acquire(False)
                    plock = self.APPROX_MMAP_PROCESS_LOCK.acquire(0)
                    if tlock and plock:
                        try:
                            with open(path_to_approx_mmap_temp, "w+b") \
                                    as mmap_file:
                                for chunk in chunks:
                                    mmap_file.write(chunk)
                            if not self.closed:
                                os.rename(path_to_approx_mmap_temp,
                                          self.path_to_approx_mmap)
                            else:
                                return
                        finally:
                            self.APPROX_MMAP_THREAD_LOCK.release()
                            try:
                                self.APPROX_MMAP_PROCESS_LOCK.release()
                            except BaseException:
                                pass
                sleep(1)  # Block before trying again
        return self._approx_index

    def get_elmo_embedder(self):
        """Gets an ElmoEmbedder of the vectors from the database."""
        meta_1_chunks = self.get_meta_chunks(1)
        meta_2_chunks = self.get_meta_chunks(2)
        if self._elmo_embedder is None:
            while True:
                if not self.setup_for_mmap:
                    self._setup_for_mmap()
                try:
                    elmo_embedder = ElmoEmbedder(
                        self.path_to_elmo_o_mmap, self.path_to_elmo_w_mmap)
                    self._elmo_embedder = elmo_embedder
                    break
                except BaseException:
                    path_to_elmo_w_mmap_temp = self.path_to_elmo_w_mmap \
                        + '.tmp'
                    path_to_elmo_o_mmap_temp = self.path_to_elmo_o_mmap \
                        + '.tmp'
                    tlock_w = self.ELMO_W_MMAP_THREAD_LOCK.acquire(False)
                    plock_w = self.ELMO_W_MMAP_PROCESS_LOCK.acquire(0)
                    tlock_o = self.ELMO_O_MMAP_THREAD_LOCK.acquire(False)
                    plock_o = self.ELMO_O_MMAP_PROCESS_LOCK.acquire(0)
                    if tlock_w and plock_w and tlock_o and plock_o:
                        try:
                            with open(path_to_elmo_w_mmap_temp, "w+b") \
                                    as mmap_file:
                                for chunk in meta_1_chunks:
                                    mmap_file.write(chunk)
                            if not self.closed:
                                os.rename(path_to_elmo_w_mmap_temp,
                                          self.path_to_elmo_w_mmap)
                            else:
                                return
                            with open(path_to_elmo_o_mmap_temp, "w+b") \
                                    as mmap_file:
                                for chunk in meta_2_chunks:
                                    mmap_file.write(chunk)
                            if not self.closed:
                                os.rename(path_to_elmo_o_mmap_temp,
                                          self.path_to_elmo_o_mmap)
                            else:
                                return
                        finally:
                            self.ELMO_W_MMAP_THREAD_LOCK.release()
                            try:
                                self.ELMO_W_MMAP_PROCESS_LOCK.release()
                            except BaseException:
                                pass
                            self.ELMO_O_MMAP_THREAD_LOCK.release()
                            try:
                                self.ELMO_O_MMAP_PROCESS_LOCK.release()
                            except BaseException:
                                pass
                sleep(1)  # Block before trying again
        return self._elmo_embedder

    def _iter(self, put_cache):
        """Yields keys and vectors for all vectors in the store."""
        try:
            db = self._db(force_new=True)
            results = db.execute(
                """
                    SELECT *
                    FROM `magnitude`
                """)
            for result in results:
                yield self._db_full_result_to_vec(result,
                                                  put_cache=put_cache)
                if self.closed:
                    return
        except Exception as e:
            if self.closed:
                pass
            else:
                raise e

    def __iter__(self):
        """Yields keys and vectors for all vectors in the store."""
        return self._iter(put_cache=True)

    def __len__(self):
        """Returns the number of vectors."""
        return self.length

    def __contains__(self, key):
        """Checks whether a key exists in the vectors"""
        return self._vector_for_key_cached(key) is not None

    def __getitem__(self, q):
        """Performs the index method when indexed."""
        if isinstance(q, slice):
            return self.index(list(range(*q.indices(self.length))),
                              return_vector=True)
        else:
            return self.index(q, return_vector=True)

    def close(self):
        """Cleans up the object"""
        self.closed = True
        while any([t.is_alive() for t in self._threads]):
            sleep(.5)
        for conn in self._all_conns:
            try:
                conn.close()
            except Exception:
                pass
        if hasattr(self, 'fd'):
            try:
                os.close(self.fd)
            except BaseException:
                pass
        try:
            self._all_vectors._mmap.close()
        except BaseException:
            pass
        try:
            del self._all_vectors
            gc.collect()
        except BaseException:
            pass
        try:
            self._approx_index.unload()
        except BaseException:
            pass
        if (hasattr(self, 'MMAP_PROCESS_LOCK') and
            hasattr(self.MMAP_PROCESS_LOCK, 'lockfile') and
                self.MMAP_PROCESS_LOCK.lockfile is not None):
            self.MMAP_PROCESS_LOCK.lockfile.close()
        if (hasattr(self, 'APPROX_MMAP_PROCESS_LOCK') and
            hasattr(self.APPROX_MMAP_PROCESS_LOCK, 'lockfile') and
                self.APPROX_MMAP_PROCESS_LOCK.lockfile is not None):
            self.APPROX_MMAP_PROCESS_LOCK.lockfile.close()

    def __del__(self):
        """ Destructor for the class """
        self.close()


class FeaturizerMagnitude(Magnitude):
    """A FeaturizerMagnitude class that subclasses Magnitude and acts as
    a way to featurize arbitrary python

    Attributes:
        number_of_values: number_of_values should be set to the
                          approximate upper-bound of the number of
                          feature values that will be looked up with query().
                          If you don't know the exact number, be conservative
                          and pick a large number, while keeping in mind the
                          bigger number_of_values is, the more memory it will
                          consume
        namespace: an optional namespace that will be prepended to each query
                   if provided
    """

    def __init__(self, number_of_values=1000000, namespace=None, **kwargs):

        self.namespace = namespace

        super(
            FeaturizerMagnitude,
            self).__init__(
            None,
            _number_of_values=number_of_values,
            _namespace=self.namespace,
            **kwargs)


class ConcatenatedMagnitude(object):

    """A ConcatenatedMagnitude class that acts as a concatenated interface
    to querying multiple magnitude objects.

    Attributes:
        *args: each arg should be a Magnitude object
    """

    def __init__(self, *args, **kwargs):
        if len(args) < 2:
            raise RuntimeError(
                "Must concatenate at least 2 Magnitude objects.")
        self.magnitudes = args
        self.dim = sum([m.dim for m in self.magnitudes])
        all_use_numpy = [m.use_numpy for m in self.magnitudes]
        if not all(use_numpy == all_use_numpy[0]
                   for use_numpy in all_use_numpy):
            raise RuntimeError(
                "All magnitude objects must have the same use_numpy value.")
        self.use_numpy = all_use_numpy[0]

    def _take(self, q, multikey, i):
        """Selects only the i'th element from the inner-most axis and
        reduces the dimensions of the tensor q by 1.
        """
        if multikey == -1:
            return q
        else:
            cut = np.take(q, [i], axis=multikey)
            result = np.reshape(cut, np.shape(cut)[0:-1]).tolist()
            return result

    def _hstack(self, l, use_numpy):
        """Horizontally stacks NumPy arrays or Python lists"""
        if use_numpy:
            return np.concatenate(l, axis=-1)
        else:
            return list(chain.from_iterable(l))

    def _dstack(self, l, use_numpy):
        """Depth stacks NumPy arrays or Python lists"""
        if use_numpy:
            return np.concatenate(l, axis=-1)
        else:
            return [self._hstack((l3[example] for l3 in l),
                                 use_numpy=use_numpy) for example in xrange(len(l[0]))]  # noqa

    @lru_cache(DEFAULT_LRU_CACHE_SIZE, ignore_unhashable_args=True)
    def query(self, q, pad_to_length=None,
              pad_left=None, truncate_left=None):
        """Handles a query of keys which could be a single key, a
        1-D list of keys, or a 2-D list of keys.
        """

        # Check if keys are specified for each concatenated model
        multikey = -1
        if isinstance(q, tuple):
            multikey = 0
        if isinstance(q, list) and isinstance(q[0], tuple):
            multikey = 1
        if (isinstance(q, list) and isinstance(q[0], list) and
                isinstance(q[0][0], tuple)):
            multikey = 2

        # Define args
        pad_to_length = pad_to_length or self.magnitudes[0].pad_to_length
        pad_left = pad_left or self.magnitudes[0].pad_left
        truncate_left = truncate_left or self.magnitudes[0].truncate_left

        # Query each model with the right set of keys
        v = [m.query(self._take(q, multikey, i)) for i, m in enumerate(
            self.magnitudes)]

        if not isinstance(q, list):  # Single key
            return self._hstack(v, self.use_numpy)
        elif isinstance(q, list) \
                and (len(q) == 0 or not isinstance(q[0], list)):  # 1D list
            return self._hstack(v, self.use_numpy)
        elif isinstance(q, list):  # 2D List
            return self._dstack(v, self.use_numpy)


class MagnitudeUtils(object):
    """A MagnitudeUtils class that contains static helper utilities."""

    @staticmethod
    def download_model(
            model,
            download_dir=os.path.expanduser('~/.magnitude/'),
            remote_path='http://magnitude.plasticity.ai/',
            _local=False):
        """ Downloads a remote Magnitude model locally (if it doesn't already
        exist) and synchronously returns the local file path once it has
        been completed """

        # Clean the inputs
        orig_model = model
        if model.endswith('.magnitude'):
            model = model[:-10]
        if model.startswith('http://') or model.startswith('https://'):
            remote_path = ''
        if model.startswith('http://magnitude.plasticity.ai/'):
            model = model.replace('http://magnitude.plasticity.ai/', '')
            remote_path = 'http://magnitude.plasticity.ai/'
        if model.startswith('https://magnitude.plasticity.ai/'):
            model = model.replace('https://magnitude.plasticity.ai/', '')
            remote_path = 'https://magnitude.plasticity.ai/'
        if not remote_path.endswith('/') and len(remote_path) > 0:
            remote_path = remote_path + '/'

        # Make the download directories
        try:
            os.makedirs(download_dir)
        except OSError:
            if not os.path.isdir(download_dir):
                raise RuntimeError("The download folder is not a folder.")

        # Local download
        local_file_name = model.replace('/', '_') + '.magnitude'
        local_file_name_tmp = model.replace('/', '_') + '.magnitude.tmp'
        remote_file_path = remote_path + model + '.magnitude'

        if not os.path.isfile(os.path.join(download_dir, local_file_name)):
            try:
                urlretrieve(
                    remote_file_path,
                    os.path.join(download_dir, local_file_name_tmp)
                )
                conn = sqlite3.connect(
                    os.path.join(
                        download_dir,
                        local_file_name_tmp))
                conn.cursor().execute("SELECT * FROM magnitude_format")
                conn.close()
                os.rename(
                    os.path.join(
                        download_dir,
                        local_file_name_tmp),
                    os.path.join(
                        download_dir,
                        local_file_name))
            except BaseException:
                if _local:
                    raise RuntimeError(
                        "The path to the Magnitude file at '" + orig_model + "' could not be found. Also failed to find a valid remote model at the following URL: " +  # noqa
                        remote_file_path)
                else:
                    raise RuntimeError(
                        "The download could not be completed. Are you sure a valid model exists at the following URL: " +  # noqa
                        remote_file_path)
        return os.path.join(download_dir, local_file_name)

    @staticmethod
    def batchify(X, y, batch_size):  # noqa: N803
        """ Creates an iterator that chunks `X` and `y` into batches
        that each contain `batch_size` elements and loops forever"""
        X_batch_generator = cycle([X[i: i + batch_size]  # noqa: N806
                                   for i in xrange(0, len(X), batch_size)])
        y_batch_generator = cycle([y[i: i + batch_size]
                                   for i in xrange(0, len(y), batch_size)])
        return izip(X_batch_generator, y_batch_generator)

    @staticmethod
    def class_encoding():
        """Creates a set of functions to add a new class, convert a
        class into an integer, and the integer back to a class."""
        class_to_int_map = {}
        int_to_class_map = None

        def add_class(c):
            global int_to_class_map
            int_to_class_map = None
            return class_to_int_map.setdefault(
                c, len(class_to_int_map))

        def class_to_int(c):
            return class_to_int_map[c]

        def int_to_class(i):
            global int_to_class_map
            if int_to_class_map is None:
                int_to_class_map = {v: k
                                    for k, v in (
                                        (
                                            hasattr(class_to_int_map, 'iteritems') and  # noqa
                                            class_to_int_map.iteritems
                                        ) or

                                        class_to_int_map.items
                                    )()}
            return int_to_class_map[i]

        return add_class, class_to_int, int_to_class

    @staticmethod
    def to_categorical(y, num_classes=None):
        """Converts a class vector (integers) to binary class matrix.
        """
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=np.float32)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    @staticmethod
    def from_categorical(categorical):
        """Converts a binary class matrix to a class vector (integers)"""
        return np.argmax(categorical, axis=1)
