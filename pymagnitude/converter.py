from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import io
import sys
import argparse
import hashlib
import lz4.frame
import tempfile
import numpy as np

from annoy import AnnoyIndex
from collections import Counter
from functools import partial
from itertools import chain, tee

try:
    from itertools import imap
except ImportError:
    imap = map
try:
    xrange
except NameError:
    xrange = range

try:
    sys.path.append(os.path.dirname(__file__) + '/third_party/internal/')
    from pymagnitude.third_party.internal.pysqlite2 import dbapi2 as sqlite3
    db = sqlite3.connect(':memory:')
    db.close()
    SQLITE_LIB = 'internal'
except BaseException:
    import sqlite3
    SQLITE_LIB = 'system'

DEFAULT_PRECISION = 7
DEFAULT_NGRAM_BEG = 3
DEFAULT_NGRAM_END = 6
SQLITE_TOKEN_SPLITTERS = ''.join([chr(i) for i in range(128)
                                  if not chr(i).isalnum()])
BOW = u"\uF000"
EOW = u"\uF000"


def try_deleting(path):
    try:
        os.remove(path)
    except BaseException:
        pass


def eprint(s):
    if __name__ == "__main__":
        sys.stderr.write(s + '\n')
        sys.stderr.flush()


def entropy(counter):
    total_count = sum(counter.values())
    probs = np.array([float(counter[bucket]) /
                      float(total_count) for bucket in counter])
    return -probs.dot(np.log2(probs))

# MD5s a file in chunks in a streaming fashion


def md5_file(path, block_size=256 * 128):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def fast_md5_file(path, block_size=256 * 128):
    md5 = hashlib.md5()
    f_size = os.path.getsize(path)
    if f_size <= 104857600:
        return md5_file(path, block_size)
    clipped_f_size = f_size - (block_size + 1)
    md5.update(str(f_size).encode('utf-8'))
    interval = 25
    seek_interval = int(float(clipped_f_size) / float(interval))
    with open(path, 'rb') as f:
        for i in range(interval):
            f.seek((i * seek_interval) % clipped_f_size)
            chunk = f.read(block_size)
            md5.update(chunk)
    return md5.hexdigest()


def char_ngrams(key, beg, end):
    return chain.from_iterable((imap(lambda ngram: ''.join(ngram), zip(
        *[key[i:] for i in xrange(j)])) for j in xrange(beg, min(len(key) + 1, end + 1))))  # noqa


def convert(input_file_path, output_file_path=None,
            precision=DEFAULT_PRECISION, subword=False,
            subword_start=DEFAULT_NGRAM_BEG,
            subword_end=DEFAULT_NGRAM_END,
            approx=False, approx_trees=None):

    files_to_remove = []
    subword = int(subword)
    approx = int(approx)

    # If no output_file_path specified, create it in a tempdir
    if output_file_path is None:
        output_file_path = os.path.join(
            tempfile.mkdtemp(),
            fast_md5_file(input_file_path) +
            '.magnitude')
        if os.path.isfile(output_file_path):
            try:
                conn = sqlite3.connect(output_file_path)
                db = conn.cursor()
                db.execute(
                    "SELECT value FROM magnitude_format WHERE key='size'") \
                    .fetchall()[0][0]
                conn.close()
                # File already exists and is functioning
                return output_file_path
            except BaseException:
                pass

    # Check args
    input_is_text = input_file_path.endswith('.txt') or \
        input_file_path.endswith('.vec')
    input_is_binary = input_file_path.endswith('.bin')
    if not input_is_text and not input_is_binary:
        exit("The input file path must be .txt, .bin, or .vec")
    if not output_file_path.endswith('.magnitude'):
        exit("The output file path file path must be .magnitude")

    # Detect GloVe format and convert to word2vec if detected
    detected_glove = False
    if input_is_text:
        with io.open(input_file_path, mode="r", encoding="utf-8",
                     errors="ignore") as ifp:
            line1 = None
            line2 = None
            while line1 is None or line2 is None:
                line = ifp.readline().strip()
                if len(line) > 0:
                    if line1 is None:
                        line1 = line
                    elif line2 is None:
                        line2 = line
            line1 = line1.replace('\t', ' ')
            line2 = line2.replace('\t', ' ')
            line1 = line1.split()
            line2 = line2.split()
            if len(line1) == len(line2):  # No header line present
                detected_glove = True
    if detected_glove:
        eprint("Detected GloVe format! Converting to word2vec format first..."
               "(this may take some time)")
        temp_file_path = os.path.join(
            tempfile.mkdtemp(), os.path.basename(input_file_path) + '.txt')
        try:
            import gensim
        except ImportError:
            raise ImportError("You need gensim >= 3.3.0 installed with pip \
                (`pip install gensim`) to convert GloVe files.")
        gensim.scripts.glove2word2vec.glove2word2vec(
            input_file_path,
            temp_file_path
        )
        input_file_path = temp_file_path
        files_to_remove.append(temp_file_path)

    # Open and load vector file
    eprint("Loading vectors... (this may take some time)")
    number_of_keys = None
    dimensions = None
    if input_is_binary:
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            raise ImportError("You need gensim >= 3.3.0 installed with pip \
                (`pip install gensim`) to convert binary files.")
        keyed_vectors = KeyedVectors.load_word2vec_format(
            input_file_path, binary=input_is_binary)
        number_of_keys = len(keyed_vectors.vectors)
        dimensions = len(keyed_vectors.vectors[0])
    else:
        # Read it manually instead of with gensim so we can stream large models
        class KeyedVectors:
            pass

        def keyed_vectors_generator():
            number_of_keys, dimensions = (None, None)
            f = io.open(input_file_path, mode="r", encoding="utf-8",
                        errors="ignore")
            first_line = True
            for line in f:
                line_split = line.strip().replace('\t', ' ').split()
                if len(line_split) == 0:
                    continue
                if first_line:
                    first_line = False
                    number_of_keys = int(line_split[0])
                    dimensions = int(line_split[1])
                    yield (number_of_keys, dimensions)
                else:
                    empty_key = len(line_split) == dimensions
                    vec_floats = line_split if empty_key else line_split[1:]
                    key = "" if empty_key else line_split[0]
                    if len(vec_floats) > dimensions:
                        key = " ".join(
                            [key] + vec_floats[0:len(vec_floats) - dimensions])
                        vec_floats = vec_floats[len(vec_floats) - dimensions:]
                    vector = np.asarray([float(elem)
                                         for elem in vec_floats])
                    yield (key, vector)
        keyed_vectors = KeyedVectors()
        kv_gen = keyed_vectors_generator()
        number_of_keys, dimensions = next(kv_gen)
        kv_gen_1, kv_gen_2 = tee(kv_gen)
        keyed_vectors.vectors = imap(lambda kv: kv[1], kv_gen_1)
        keyed_vectors.index2word = imap(lambda kv: kv[0], kv_gen_2)

    eprint("Found %d key(s)" % number_of_keys)
    eprint("Each vector has %d dimension(s)" % dimensions)

    # Connect to magnitude datastore
    try_deleting(output_file_path)
    try_deleting(output_file_path + "-shm")
    try_deleting(output_file_path + "-wal")
    conn = sqlite3.connect(output_file_path)
    files_to_remove.append(output_file_path + "-shm")
    files_to_remove.append(output_file_path + "-wal")
    db = conn.cursor()

    # Make the database fast
    conn.isolation_level = None
    db.execute("PRAGMA synchronous = OFF;")
    db.execute("PRAGMA default_synchronous = OFF;")
    db.execute("PRAGMA journal_mode = WAL;")
    db.execute("PRAGMA count_changes = OFF;")

    # Create table structure
    eprint("Creating magnitude format...")
    db.execute("DROP TABLE IF EXISTS `magnitude`;")
    db.execute("""
        CREATE TABLE `magnitude` (
            key TEXT COLLATE NOCASE,
            """ +
               ",\n".join([("dim_%d INTEGER" % i) for i in range(dimensions)]) +
               """
        );
    """)
    db.execute("""
        CREATE TABLE `magnitude_format` (
            key TEXT COLLATE NOCASE,
            value INTEGER
        );
    """)
    if subword:
        db.execute("""
            CREATE VIRTUAL TABLE `magnitude_subword`
            USING fts3(
                char_ngrams,
                num_ngrams
            );
        """)
    if approx:
        db.execute("""
            CREATE TABLE `magnitude_approx` (
                trees INTEGER,
                index_file BLOB
            );
        """)

    # Create annoy index
    approx_index = None
    if approx:
        approx_index = AnnoyIndex(dimensions)

    # Write vectors
    eprint("Writing vectors... (this may take some time)")
    insert_query = """
        INSERT INTO `magnitude`(
            key,
            """ + \
        ",\n".join([("dim_%d" % i) for i in range(dimensions)]) \
        + """)
        VALUES (
            """ + \
        (",\n".join(["?"] * (dimensions + 1))) \
        + """
        );
    """
    insert_subword_query = """
        INSERT INTO `magnitude_subword`(
            char_ngrams,
            num_ngrams
        )
        VALUES (
            ?, ?
        );
    """
    counters = [Counter() for i in range(dimensions)]
    key_vectors_iterable = zip(keyed_vectors.index2word, keyed_vectors.vectors)
    progress = -1
    db.execute("BEGIN;")
    for i, (key, vector) in enumerate(key_vectors_iterable):
        current_progress = int((float(i) / float(number_of_keys)) * 100)
        if current_progress > progress:
            progress = current_progress
            eprint("%d%% completed" % progress)
        if i % 100000 == 0:
            db.execute("COMMIT;")
            db.execute("BEGIN;")
        vector = vector / np.linalg.norm(vector)
        epsilon = np.random.choice(
            [-1.0 / (10**precision), 1.0 / (10**precision)], dimensions)
        vector = epsilon if np.isnan(vector).any() else vector
        for d, v in enumerate(vector):
            counters[d][int(v * 100)] += 1
        db.execute(insert_query, (key,) + tuple(int(round(v * (10**precision)))
                                                for v in vector))
        if subword:
            ngrams = set(
                (n.lower() for n in char_ngrams(
                    BOW + key + EOW,
                    subword_start,
                    subword_end)))
            num_ngrams = len(ngrams) * 4
            ngrams = set((n for n in ngrams if not any(
                [c in SQLITE_TOKEN_SPLITTERS for c in n])))
            db.execute(insert_subword_query,
                       (" ".join(ngrams), num_ngrams))
        if approx:
            approx_index.add_item(i, vector)
    eprint("Committing written vectors... (this may take some time)")
    db.execute("COMMIT;")

    # Figure out which dimensions have the most entropy
    entropies = [(d, entropy(counter)) for d, counter in enumerate(counters)]
    entropies.sort(key=lambda e: e[1], reverse=True)
    for e in entropies:
        eprint("Entropy of dimension %d is %f" % (e[0], e[1]))
    highest_entropy_dimensions = [e[0] for e in entropies]

    # Writing metadata
    insert_format_query = """
        INSERT INTO `magnitude_format`(
            key,
            value
        )
        VALUES (
            ?, ?
        );
    """

    db.execute(insert_format_query, ('size', number_of_keys))
    db.execute(insert_format_query, ('dim', dimensions))
    db.execute(insert_format_query, ('precision', precision))
    if subword:
        db.execute(insert_format_query, ('subword', subword))
        db.execute(insert_format_query, ('subword_start', subword_start))
        db.execute(insert_format_query, ('subword_end', subword_end))
    if approx:
        if approx_trees is None:
            approx_trees = max(50, int((number_of_keys / 3000000.0) * 50.0))
        db.execute(insert_format_query, ('approx', approx))
        db.execute(insert_format_query, ('approx_trees', approx_trees))
    for d in highest_entropy_dimensions:
        db.execute(insert_format_query, ('entropy', d))

    # Create indicies
    eprint("Creating search index... (this may take some time)")
    db.execute("CREATE INDEX `magnitude_key_idx` ON `magnitude` (key);")
    for i in highest_entropy_dimensions[0:1]:
        eprint("Creating spatial search index for dimension %d "
               "(it has high entropy)... (this may take some time)" % i)
        db.execute("""
            CREATE INDEX `magnitude_dim_%d_idx` ON `magnitude` (dim_%d);
        """ % (i, i))

    # Write approximate index to the database
    if approx:
        eprint("Creating approximate nearest neighbors index... \
(this may take some time)")
        approx_index.build(approx_trees)
        approx_index_file_path = os.path.join(
            tempfile.mkdtemp(),
            fast_md5_file(input_file_path) + '.ann')
        eprint("Dumping approximate nearest neighbors index... \
(this may take some time)")
        approx_index.save(approx_index_file_path)
        eprint("Compressing approximate nearest neighbors index... \
(this may take some time)")
        chunk_size = 104857600
        full_size = os.path.getsize(approx_index_file_path)
        insert_approx_query = """
            INSERT INTO magnitude_approx(trees, index_file) VALUES (?, ?);
        """
        with open(approx_index_file_path, 'rb') as ifh, \
                lz4.frame.LZ4FrameCompressor() as compressor:
            for i, chunk in enumerate(iter(partial(ifh.read, chunk_size), b'')):
                if i == 0:
                    chunk = compressor.begin() + compressor.compress(chunk)
                else:
                    chunk = compressor.compress(chunk)
                eprint(str((ifh.tell() / float(full_size)) * 100.0) + "%")
                if len(chunk) > 0:
                    db.execute(insert_approx_query,
                               (approx_trees, sqlite3.Binary(chunk)))
            chunk = compressor.flush()
            if len(chunk) > 0:
                db.execute(insert_approx_query,
                           (approx_trees, sqlite3.Binary(chunk)))
        files_to_remove.append(approx_index_file_path)

    # VACUUM
    eprint("Vacuuming to save space... (this may take some time)")
    db.execute("VACUUM;")

    # Restore safe database settings
    db.execute("PRAGMA synchronous = FULL;")
    db.execute("PRAGMA default_synchronous = FULL;")
    db.execute("PRAGMA journal_mode = DELETE;")
    db.execute("PRAGMA count_changes = ON;")

    # Clean up connection
    conn.commit()
    conn.close()

    # Clean up
    if len(files_to_remove) > 0:
        eprint("Cleaning up temporary files...")
        for file_to_remove in files_to_remove:
            try_deleting(file_to_remove)

    # Print success
    eprint("Successfully converted '%s' to '%s'!" %
           (input_file_path, output_file_path))

    return output_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="input file (.txt, bin, or .vec) or folder",
        type=str, required=True)
    parser.add_argument(
        "-o", "--output", help="output file (.magnitude) or folder", type=str,
        required=True)
    parser.add_argument(
        "-p", "--precision", type=int, default=DEFAULT_PRECISION,
        help=("decimal precision, a lower number saves space (default: %d)"
              % DEFAULT_PRECISION))
    parser.add_argument(
        "-s", "--subword-off", action='store_true',
        help="don't enrich file with subword info for out-of-vocabulary words \
(saves space)")
    parser.add_argument(
        "-w", "--window", help=("subword character n-gram min,max window \
(default: %d,%d)"
                                % (DEFAULT_NGRAM_BEG, DEFAULT_NGRAM_END)),
        type=str, default=("%d,%d" % (DEFAULT_NGRAM_BEG, DEFAULT_NGRAM_END)))
    parser.add_argument(
        "-a", "--approx", action='store_true',
        help="build an index for approximate nearest neighbors for the \
`most_similar_approx()` function. (uses a lot of space, but approximate most \
similar queries are faster)")
    parser.add_argument(
        "-t", "--trees", type=int,
        help=("number of trees for the approximate nearest neighbors index. \
If not provided, this will be determined automatically. (higher number uses \
more space, but makes approximate most similar queries more accurate)"))
    args = parser.parse_args()

    input_file_path = os.path.expanduser(args.input)
    output_file_path = os.path.expanduser(args.output)
    precision = args.precision
    subword = not(args.subword_off)
    subword_start = int(args.window.split(",")[0])
    subword_end = int(args.window.split(",")[1])
    approx = args.approx
    approx_trees = args.trees if hasattr(args, 'trees') else None

    if os.path.isdir(input_file_path) and os.path.isdir(output_file_path):
        for file in os.listdir(input_file_path):
            if (file.endswith(".txt") or
                file.endswith(".bin") or
                    file.endswith(".vec")):
                ext = '.magnitude'
                split_file_name = os.path.basename(file).split(".")
                output_file = ".".join(split_file_name[0:-1]) + ext
                eprint("Creating: %s" %
                       os.path.join(output_file_path, output_file))
                convert(os.path.join(input_file_path, file),
                        os.path.join(output_file_path, output_file),
                        precision=precision, subword=subword,
                        subword_start=subword_start,
                        subword_end=subword_end,
                        approx=approx, approx_trees=approx_trees)
    else:
        convert(input_file_path, output_file_path,
                precision=precision, subword=subword,
                subword_start=subword_start,
                subword_end=subword_end,
                approx=approx, approx_trees=approx_trees)
