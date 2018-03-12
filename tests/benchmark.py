#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gc
import random
import sys
import time
import tempfile

from functools import partial
from sys import platform as _platform
from random import seed, randint
from itertools import islice
from gensim.models import KeyedVectors

from pymagnitude import Magnitude


BIN_PATH = sys.argv[1]
MAGNITUDE_PATH = sys.argv[2]
REPEATS = 1

################################
# Utilities
################################


def get_size(obj, seen=None):
    """Recursively finds size of objects
       Source: https://goshippo.com/blog/measure-real-size-any-python-object/
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        try:
            size += get_size(obj.__dict__, seen)
        except BaseException:
            pass
    elif hasattr(obj, '__iter__') and not isinstance(obj,
                                                     (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def _clear_memory_buffers():
    # Various methods of clearing the memory buffers/caches
    if _platform == "darwin":
        os.system("purge")
    else:
        os.system("sync && echo 3 > /proc/sys/vm/drop_caches")


def _clear_mmap():
    os.system("rm -rf " + os.path.join(tempfile.gettempdir(), '*.magmmap'))
    os.system("rm -rf " + os.path.join(tempfile.gettempdir(), '*.magmmap*'))


def run_benchmark(name, func_1, func_2=None,
                  repeat=REPEATS, clear_mmap=True,
                  clear_memory_buffers=True,
                  args_1=[], args_2=[]):
    def _start_func(timer_start):
        timer_start[0] = time.time()
    print("==================================")
    print("Running benchmark:")
    print("----------------------------------")
    print(name)
    print("----------------------------------")
    print("Running '" + func_1.__name__ + "' %d time(s)" % repeat)
    times = []
    return_1 = None
    return_2 = None
    for i in xrange(repeat):
        s = [None]
        if clear_memory_buffers:
            _clear_memory_buffers()
        if clear_mmap:
            _clear_mmap()
        return_1 = func_1(partial(_start_func, s), *args_1)
        times.append(time.time() - s[0])
        gc.collect()
    time1 = sum(times) / len(times)
    print("'%s' ran in %.6f second(s)" % (func_1.__name__, time1))
    if func_2 is not None:
        print("Running '" + func_2.__name__ + "' %d time(s)" % repeat)
        times = []
        for i in xrange(repeat):
            s = [None]
            if clear_memory_buffers:
                _clear_memory_buffers()
            if clear_mmap:
                _clear_mmap()
            return_2 = func_2(partial(_start_func, s), *args_2)
            times.append(time.time() - s[0])
            gc.collect()
        time2 = sum(times) / len(times)
        print("'%s' ran in %.6f second(s)" % (func_2.__name__, time2))
    print("==================================")
    return return_1, return_2


def run_memory_benchmark(name, obj_1, obj_2):
    divider = float((1024**2))
    print("==================================")
    print("Running memory benchmark:")
    print("----------------------------------")
    print(name)
    print("----------------------------------")
    print("Recursively calculating memory usage of '" + str(obj_1) + "'")
    print("'%s' is using %.6f MB" % (str(obj_1), get_size(obj_1) / divider))
    print("Recursively calculating memory usage of '" + str(obj_2) + "'")
    print("'%s' is using %.6f MB" % (str(obj_2), get_size(obj_2) / divider))
    print("==================================")


def run_integrity_checks(vectors_magnitude, vectors_gensim,
                         small_n=10, big_n=1000):
    def _floats_equal(a, b):
        return abs(a - b) < .00001
    seed(0)  # Reproducible results
    small_i = [randint(0, len(vectors_magnitude) - 1) for i in range(small_n)]
    big_i = [randint(0, len(vectors_magnitude) - 1) for i in range(big_n)]
    small_keys = [vectors_magnitude[i][0] for i in small_i]
    big_keys = [vectors_magnitude[i][0] for i in big_i]
    print("==================================")
    print("Running integrity checks to make sure file is valid...")
    print("----------------------------------")
    print("Checking lengths are equal...")
    print(len(vectors_magnitude) == len(vectors_gensim.index2word),
          len(vectors_magnitude), len(vectors_gensim.index2word))
    print("Checking dimensions are equal...")
    shape_1 = next(iter(vectors_magnitude))[1].shape
    shape_2 = vectors_gensim.vectors[0].shape
    print(shape_1 == shape_2, shape_1, shape_2)
    print("Checking %d random similarity calls..." % big_n)
    big_key_it = iter(big_keys)
    for i, key1 in enumerate(big_key_it):
        if i % 10 == 0:
            print("Progress: %d/%d" % (i, len(big_keys) / 2))
        key2 = next(big_key_it)
        result_1 = vectors_magnitude.similarity(key1, key2)
        result_2 = vectors_gensim.similarity(key1, key2)
        if not _floats_equal(result_1, result_2):
            print("Mismatched similarities (%f, %f) for keys: (%s, %s)" %
                  (result_1, result_2, key1, key2))
    print("Checking analogy...")
    results_1 = vectors_magnitude.most_similar(positive=["king", "woman"],
                                               negative=["man"])
    results_2 = vectors_gensim.most_similar(positive=["king", "woman"],
                                            negative=["man"])
    print(results_1)
    print(results_2)
    print("Checking 3cosmul analogy...")
    results_1 = vectors_magnitude.most_similar_cosmul(
        positive=["king", "woman"],
        negative=["man"])
    results_2 = vectors_gensim.most_similar_cosmul(
        positive=["king", "woman"],
        negative=["man"])
    print(results_1)
    print(results_2)
    print("Checking %d random most_similar calls..." % small_n)
    for key in small_keys:
        results_1 = vectors_magnitude.most_similar(key)
        results_2 = vectors_gensim.most_similar(key)
        if len(results_1) == len(results_2):
            broken = False
            for result_1, result_2 in zip(results_1, results_2):
                if result_1[0] != result_2[0] or \
                        not _floats_equal(result_1[1], result_2[1]):
                    broken = True
            if broken:
                print("--------------------------------")
                print("Mismatched results for key: %s" % key)
                print(results_1)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                print(results_2)
                print("--------------------------------")
            else:
                print(not broken)
        else:
            print("Lengths (%d, %d) not equal for key %s.",
                  len(results_1), len(results_2), key)
    print("==================================")

################################
# Model Creators
################################


def create_gensim():
    keyed_vectors = KeyedVectors.load_word2vec_format(BIN_PATH,
                                                      binary=True)
    return keyed_vectors


def create_magnitude(case_insensitive=True, eager=False, **kwargs):
    vectors = Magnitude(
        MAGNITUDE_PATH,
        case_insensitive=case_insensitive,
        eager=eager,
        **kwargs)
    return vectors

################################
# Tests
################################


def initial_load_magnitude(start):
    start()
    vectors = create_magnitude()


def intitial_load_gensim(start):
    start()
    vectors = create_gensim()


def cold_query_magnitude(start):
    vectors = create_magnitude()
    keys = ["cat", "parrot", "monkey", "buffalo",
            "lion", "snake", "lizard", "tiger", "alligator",
            "ostrich", "elephant", "whale", "dolphin", "fish",
            "jellyfish", "fox", "rabbit", "badger", "bear",
            "chicken", "kangaroo", "deer", "duck", "shark"]
    start()
    vectors.query("dog")


def cold_query_gensim(start):
    vectors = create_gensim()
    start()
    vectors.get_vector("dog")
    return vectors


def warm_query_magnitude(start, vectors):
    start()
    vectors.query("dog")
    return vectors


def warm_query_gensim(start, vectors):
    start()
    vectors.get_vector("dog")
    return vectors


def cold_multi_query_magnitude(start):
    vectors = create_magnitude()
    start()
    vectors.query(["dog", "cat", "parrot", "monkey", "buffalo",
                   "lion", "snake", "lizard", "tiger", "alligator",
                   "ostrich", "elephant", "whale", "dolphin", "fish",
                   "jellyfish", "fox", "rabbit", "badger", "bear",
                   "chicken", "kangaroo", "deer", "duck", "shark"])


def cold_multi_query_gensim(start):
    vectors = create_gensim()
    start()
    vectors.get_vector("dog")
    vectors.get_vector("cat")
    vectors.get_vector("parrot")
    vectors.get_vector("monkey")
    vectors.get_vector("buffalo")
    vectors.get_vector("lion")
    vectors.get_vector("snake")
    vectors.get_vector("lizard")
    vectors.get_vector("tiger")
    vectors.get_vector("alligator")
    vectors.get_vector("ostrich")
    vectors.get_vector("elephant")
    vectors.get_vector("whale")
    vectors.get_vector("dolphin")
    vectors.get_vector("fish")
    vectors.get_vector("jellyfish")
    vectors.get_vector("fox")
    vectors.get_vector("rabbit")
    vectors.get_vector("badger")
    vectors.get_vector("bear")
    vectors.get_vector("chicken")
    vectors.get_vector("kangaroo")
    vectors.get_vector("deer")
    vectors.get_vector("duck")
    vectors.get_vector("shark")


def warm_multi_query_magnitude(start, vectors):
    start()
    vectors.query(["dog", "cat", "parrot", "monkey", "buffalo",
                   "lion", "snake", "lizard", "tiger", "alligator",
                   "ostrich", "elephant", "whale", "dolphin", "fish",
                   "jellyfish", "fox", "rabbit", "badger", "bear",
                   "chicken", "kangaroo", "deer", "duck", "shark"])


def warm_multi_query_gensim(start, vectors):
    start()
    vectors.get_vector("dog")
    vectors.get_vector("cat")
    vectors.get_vector("parrot")
    vectors.get_vector("monkey")
    vectors.get_vector("buffalo")
    vectors.get_vector("lion")
    vectors.get_vector("snake")
    vectors.get_vector("lizard")
    vectors.get_vector("tiger")
    vectors.get_vector("alligator")
    vectors.get_vector("ostrich")
    vectors.get_vector("elephant")
    vectors.get_vector("whale")
    vectors.get_vector("dolphin")
    vectors.get_vector("fish")
    vectors.get_vector("jellyfish")
    vectors.get_vector("fox")
    vectors.get_vector("rabbit")
    vectors.get_vector("badger")
    vectors.get_vector("bear")
    vectors.get_vector("chicken")
    vectors.get_vector("kangaroo")
    vectors.get_vector("deer")
    vectors.get_vector("duck")
    vectors.get_vector("shark")


def first_most_similar_magnitude(start):
    vectors = create_magnitude()
    start()
    vectors.most_similar("cat")
    return vectors


def first_most_similar_approx_magnitude(start, close=True):
    vectors = create_magnitude()
    start()
    vectors.most_similar_approx("cat", effort=1.0)
    if close:
        vectors.close()
    return vectors


def first_most_similar_gensim(start):
    vectors = create_gensim()
    start()
    vectors.most_similar("cat")
    return vectors


def subsequent_most_similar_magnitude(start, vectors):
    start()
    vectors.most_similar("dog")


def subsequent_most_similar_approx_magnitude(start, vectors, effort):
    random_key = vectors.index(
        random.randint(
            0,
            len(vectors) - 1),
        return_vector=False)
    start()
    vectors.most_similar_approx(random_key, effort=effort)


def subsequent_most_similar_gensim(start, vectors):
    start()
    vectors.most_similar("dog")


def warm_most_similar_magnitude(start, vectors):
    start()
    vectors.most_similar("cat")


def warm_most_similar_approx_magnitude(start, vectors):
    start()
    vectors.most_similar_approx("cat", effort=1.0)


def warm_most_similar_gensim(start, vectors):
    start()
    vectors.most_similar("cat")


################################
# Run Benchmarks
################################
if __name__ == "__main__":
    _clear_memory_buffers()
    _clear_mmap()

    vectors_magnitude = create_magnitude(case_insensitive=False, eager=False)
    vectors_gensim = create_gensim()
    run_integrity_checks(vectors_magnitude, vectors_gensim)
    vectors_magnitude.close()
    del vectors_magnitude
    del vectors_gensim

    run_benchmark("Initial Load Time",
                  initial_load_magnitude,
                  intitial_load_gensim,
                  repeat=1)

    run_benchmark("Cold query single key",
                  cold_query_magnitude,
                  cold_query_gensim,
                  repeat=1)

    vectors_magnitude = create_magnitude()
    warm_query_magnitude(lambda: None, vectors_magnitude)
    vectors_gensim = create_gensim()
    warm_query_gensim(lambda: None, vectors_gensim)
    run_benchmark("Warm query single key",
                  func_1=warm_query_magnitude, args_1=(vectors_magnitude,),
                  func_2=warm_query_gensim, args_2=(vectors_gensim,),
                  repeat=100
                  )
    vectors_magnitude.close()
    del vectors_magnitude
    del vectors_gensim

    run_benchmark("Cold query multiple key",
                  cold_multi_query_magnitude,
                  cold_multi_query_gensim,
                  repeat=1)

    gc.collect()
    vectors_magnitude = create_magnitude()
    warm_multi_query_magnitude(lambda: None, vectors_magnitude)
    run_benchmark("Warm query multiple key (magnitude)",
                  func_1=warm_multi_query_magnitude, args_1=(
                      vectors_magnitude,),
                  repeat=100,
                  clear_memory_buffers=False
                  )
    vectors_magnitude.close()
    del vectors_magnitude
    gc.collect()
    vectors_gensim = create_gensim()
    warm_multi_query_gensim(lambda: None, vectors_gensim)
    run_benchmark("Warm query multiple key (gensim)",
                  func_1=warm_multi_query_gensim, args_1=(vectors_gensim,),
                  repeat=100,
                  clear_memory_buffers=False
                  )
    del vectors_gensim

    run_benchmark("First most similar search (worst case)",
                  first_most_similar_magnitude,
                  first_most_similar_gensim,
                  repeat=1)

    vectors_magnitude = create_magnitude(eager=True, blocking=True)
    time.sleep(10)
    vectors_magnitude.close()
    del vectors_magnitude
    run_benchmark("First most similar search (average case) (magnitude)",
                  func_1=first_most_similar_magnitude,
                  repeat=1,
                  clear_mmap=False,
                  clear_memory_buffers=False
                  )
    gc.collect()
    vectors_gensim = create_gensim()
    time.sleep(10)
    del vectors_gensim
    run_benchmark("First most similar search (average case) (gensim)",
                  func_1=first_most_similar_gensim,
                  repeat=1,
                  clear_mmap=False,
                  clear_memory_buffers=False
                  )

    vectors_magnitude = first_most_similar_magnitude(lambda: None)
    time.sleep(10)
    run_benchmark("Subsequent most similar search (magnitude)",
                  func_1=subsequent_most_similar_magnitude,
                  args_1=(vectors_magnitude,),
                  repeat=1,
                  clear_mmap=False,
                  clear_memory_buffers=False
                  )
    vectors_magnitude.close()
    del vectors_magnitude
    gc.collect()
    vectors_gensim = first_most_similar_gensim(lambda: None)
    time.sleep(10)
    run_benchmark("Subsequent most similar search (gensim)",
                  func_1=subsequent_most_similar_gensim,
                  args_1=(vectors_gensim,),
                  repeat=10,
                  clear_mmap=False,
                  clear_memory_buffers=False
                  )
    del vectors_gensim

    vectors_magnitude = first_most_similar_magnitude(lambda: None)
    time.sleep(10)
    run_benchmark("Warm most similar search (magnitude)",
                  func_1=warm_most_similar_magnitude,
                  args_1=(vectors_magnitude,),
                  repeat=10,
                  clear_mmap=False,
                  clear_memory_buffers=False
                  )
    vectors_magnitude.close()
    del vectors_magnitude
    gc.collect()
    vectors_gensim = first_most_similar_gensim(lambda: None)
    time.sleep(10)
    run_benchmark("Warm most similar search (gensim)",
                  func_1=warm_most_similar_gensim,
                  args_1=(vectors_gensim,),
                  repeat=10,
                  clear_mmap=False,
                  clear_memory_buffers=False
                  )
    del vectors_gensim

    run_benchmark("First most similar approx search (worst case)",
                  func_1=first_most_similar_approx_magnitude,
                  repeat=1)

    vectors_magnitude = create_magnitude(eager=True, blocking=True)
    vectors_magnitude.get_approx_index()
    time.sleep(10)
    vectors_magnitude.close()
    del vectors_magnitude
    run_benchmark(
        "First most similar approx (effort = 1.0) search (average case) (magnitude)",
        func_1=first_most_similar_approx_magnitude,
        repeat=1,
        clear_mmap=False,
        clear_memory_buffers=False)

    vectors_magnitude = create_magnitude(eager=True, blocking=True)
    vectors_magnitude.get_approx_index()
    time.sleep(10)
    run_benchmark(
        "Subsequent most similar approx (effort = 1.0) search (magnitude)",
        func_1=subsequent_most_similar_approx_magnitude,
        args_1=(
            vectors_magnitude,
            1.0),
        repeat=1000,
        clear_mmap=False,
        clear_memory_buffers=False)
    vectors_magnitude.close()
    del vectors_magnitude

    vectors_magnitude = create_magnitude(eager=True, blocking=True)
    vectors_magnitude.get_approx_index()
    time.sleep(10)
    run_benchmark(
        "Subsequent most similar approx (effort = 0.1) search (magnitude)",
        func_1=subsequent_most_similar_approx_magnitude,
        args_1=(
            vectors_magnitude,
            0.1),
        repeat=1000,
        clear_mmap=False,
        clear_memory_buffers=False)
    vectors_magnitude.close()
    del vectors_magnitude

    vectors_magnitude = first_most_similar_approx_magnitude(
        lambda: None, close=False)
    time.sleep(10)
    run_benchmark("Warm most similar approx (effort = 1.0) search (magnitude)",
                  func_1=warm_most_similar_approx_magnitude,
                  args_1=(vectors_magnitude,),
                  repeat=100,
                  clear_mmap=False,
                  clear_memory_buffers=False
                  )
    vectors_magnitude.close()
    del vectors_magnitude

    vectors_magnitude = create_magnitude(
        case_insensitive=False, lazy_loading=100)
    vectors_gensim = create_gensim()
    run_memory_benchmark("Initial RAM Utilization",
                         vectors_magnitude, vectors_gensim)

    for key, vector in islice(vectors_magnitude, 100):
        vectors_magnitude.query(key)
        vectors_gensim.get_vector(key)
    run_memory_benchmark("Initial RAM Utilization + 100 keys",
                         vectors_magnitude, vectors_gensim)

    subsequent_most_similar_magnitude(lambda: None, vectors_magnitude)
    subsequent_most_similar_gensim(lambda: None, vectors_gensim)
    run_memory_benchmark(
        "Initial RAM Utilization + 100 keys + similarity search",
        vectors_magnitude, vectors_gensim)
    vectors_magnitude.close()
    del vectors_magnitude
    del vectors_gensim
