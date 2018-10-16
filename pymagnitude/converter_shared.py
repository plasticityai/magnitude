import hashlib
import numpy as np
import os

from itertools import islice, chain
try:
    from itertools import imap
except ImportError:
    imap = map
try:
    xrange
except NameError:
    xrange = range


DEFAULT_PRECISION = 7
DEFAULT_NGRAM_BEG = 3
DEFAULT_NGRAM_END = 6
SQLITE_TOKEN_SPLITTERS = ''.join([chr(i) for i in range(128)
                                  if not chr(i).isalnum()])
BOW = u"\uF000"
EOW = u"\uF000"
CONVERTER_VERSION = 2


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


def norm_matrix(m):
    return m / np.linalg.norm(m, axis=1).reshape((-1, 1))


def norm_elmo(e):
    for i in range(e.shape[0]):
        e[i, :, :] = norm_matrix(e[i, :, :])


def ibatch(iterable, size):
    sourceiter = iter(iterable)
    while True:
        batchiter = islice(sourceiter, size)
        yield chain([next(batchiter)], batchiter)
