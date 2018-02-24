<div align="center"><img src="https://gitlab.com/Plasticity/magnitude/raw/master/images/magnitude.png" alt="magnitude" height="50"></div>

## <div align="center">Magnitude: a fast, simple vector embedding utility library<br /><br />[![pipeline status](https://gitlab.com/Plasticity/magnitude/badges/master/pipeline.svg)](https://gitlab.com/Plasticity/magnitude/commits/master)&nbsp;&nbsp;&nbsp;[![PyPI version](https://badge.fury.io/py/pymagnitude.svg)](https://pypi.python.org/pypi/pymagnitude/)&nbsp;&nbsp;&nbsp;[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://gitlab.com/Plasticity/magnitude/blob/master/LICENSE.txt)&nbsp;&nbsp;&nbsp;&nbsp;[![Python version](https://img.shields.io/pypi/pyversions/pymagnitude.svg)](https://pypi.python.org/pypi/pymagnitude/)</div>
A feature-packed Python package and vector storage file format for utilizing vector embeddings in machine learning models in a fast, efficient, and simple manner developed by [Plasticity](https://www.plasticity.ai/). It is primarily intended to be a faster alternative to [Gensim](https://radimrehurek.com/gensim/), but can be used as a generic key-vector store for domains outside NLP.

## Installation
You can install this package with `pip`:
```python
pip install pymagnitude # Python 2.7
pip3 install pymagnitude # Python 3
```

## Motivation
Vector space embedding models have become increasingly common in machine learning and traditionally have been popular for natural language processing applications. A fast, lightweight tool to consume these large vector space embedding models efficiently is lacking.

The Magnitude file format (`.magnitude`) for vector embeddings is intended to be a more efficient universal vector embedding format that allows for lazy-loading for faster cold starts in development, LRU memory caching for performance in production, multiple key queries, direct featurization to the inputs for a neural network, performant similiarity calculations, and other nice to have features for edge cases like handling out-of-vocabulary keys or misspelled keys and concatenating multiple vector models together. It also is intended to work with large vector models that may not fit in memory.

It uses [SQLite](http://www.sqlite.org), a fast, popular embedded database, as its underlying data store. It uses indexes for fast key lookups as well as uses memory mapping, SIMD instructions, and spatial indexing for fast similarity search in the vector space off-disk with good memory performance even between multiple processes. Moreover, memory maps are cached between runs so even after closing a process, speed improvements are reaped.

## Benchmarks and Features

| **Metric**                                                                                                                                            | **Gensim**  | **Magnitude Light**   | **Magnitude Medium** | **Magnitude Heavy** |
| ------------------------------------------------------------------------------------------------------------------------                              | :---------: | :-------------------: | :------------------: | :-----------------: |
| Initial load time                                                                                                                                     | 70.26s      | **0.7210s**           | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| Cold single key query                                                                                                                                 | **0.0001s** | **0.0001s**           | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| Warm single key query <br /><sup>*(same key as cold query)*</sup>                                                                                     | 0.0044s     | **0.00004s**          | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| Cold multiple key query <br /><sup>*(n=25)*</sup>                                                                                                     | 3.0050s     | **0.0442s**           | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| Warm multiple key query <br /><sup>*(n=25) (same keys as cold query)*</sup>                                                                           | **0.0001s** | **0.00004s**          | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| First `most_similar` search query <br /><sup>*(n=10) (worst case)*</sup>                                                                              | **18.493s** | 247.05s               | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| First `most_similar` search query <br /><sup>*(n=10) (average case) (w/ disk persistent cache)*</sup>                                                 | 18.917s     | **1.8217s**           | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| Subsequent `most_similar` search <br /><sup>*(n=10) (different key than first query)*</sup>                                                           | 0.2546s     | **0.2434s**           | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| Warm subsequent `most_similar` search <br /><sup>*(n=10) (same key as first query)*</sup>                                                             | 0.2374s     | **0.00004s**          | **0.00004s**         | **0.00004s**        |
| First `most_similar_approx` search query <br /><sup>*(n=10, effort=1.0) (worst case)*</sup>                                                           | N/A         | N/A                   | N/A                  | **29.610s**         |
| First `most_similar_approx` search query <br /><sup>*(n=10, effort=1.0) (average case) (w/ disk persistent cache)*</sup>                              | N/A         | N/A                   | N/A                  | **0.9155s**         |
| Subsequent `most_similar_approx` search <br /><sup>*(n=10, effort=1.0) (different key than first query)*</sup>                                        | N/A         | N/A                   | N/A                  | **0.1873s**         |
| Subsequent `most_similar_approx` search <br /><sup>*(n=10, effort=0.1) (different key than first query)*</sup>                                        | N/A         | N/A                   | N/A                  | **0.0199s**         |
| Warm subsequent `most_similar_approx` search <br /><sup>*(n=10, effort=1.0) (same key as first query)*</sup>                                          | N/A         | N/A                   | N/A                  | **0.00004s**        |
| File size                                                                                                                                             | **3.64GB**  | 4.21GB                | 5.29GB               | 10.74GB             |
| Process memory (RAM) utilization                                                                                                                      | 4.875GB     | **18KB**              | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| Process memory (RAM) utilization after 100 key queries                                                                                                | 4.875GB     | **168KB**             | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| Process memory (RAM) utilization after 100 key queries + similarity search                                                                            | 8.228GB     | **342KB**<sup>2</sup> | ━&nbsp;<sup>1</sup>  | ━&nbsp;<sup>1</sup> |
| Integrity checks and tests                                                                                                                            | ✅           | ✅                     | ✅                    | ✅                   |
| Universal format between word2vec (`.txt`, `.bin`), GloVE (`.txt`), and fastText (`.vec`) with converter utility                                      | ❌           | ✅                     | ✅                    | ✅                   |
| Simple, Pythonic interface                                                                                                                            | ❌           | ✅                     | ✅                    | ✅                   |
| Few dependencies                                                                                                                                      | ❌           | ✅                     | ✅                    | ✅                   |
| Support for larger than memory memory models                                                                                                          | ❌           | ✅                     | ✅                    | ✅                   |
| Lazy loading whenever possible for speed and performance                                                                                              | ❌           | ✅                     | ✅                    | ✅                   |
| Optimized for `threading` and `multiprocessing`                                                                                                       | ❌           | ✅                     | ✅                    | ✅                   |
| Bulk and multiple key lookup with padding, truncation, placeholder, and featurization support                                                         | ❌           | ✅                     | ✅                    | ✅                   |
| Concatenting multiple vector models together                                                                                                          | ❌           | ✅                     | ✅                    | ✅                   |
| Basic out-of-vocabulary key lookup <br /><sup>(character n-gram feature hashing)</sup>                                                                | ❌           | ✅                     | ✅                    | ✅                   |
| Advanced out-of-vocabulary key lookup with support for misspellings <br /><sup>(character n-gram feature hashing to similar in-vocabulary keys)</sup> | ❌           | ❌                     | ✅                    | ✅                   |
| Approximate most similar search with an [annoy](#other-notable-projects) index                                                                        | ❌           | ❌                     | ❌                    | ✅                   |
| Built-in training for new models                                                                                                                      | ✅           | ❌                     | ❌                    | ❌                   |


<sup>1: *same value as previous column*</sup><br />
<sup>2: *uses `mmap` to read from disk, so the OS will still allocate pages of memory when memory is available, but it can be shared between processes and isn't managed within each process for extremely large files which is a performance win*</sup><br/>
<sup>\*: All [benchmarks](https://gitlab.com/Plasticity/magnitude/blob/master/tests/benchmark.py) were performed on the Google News pre-trained word vectors (`GoogleNews-vectors-negative300.bin`) with a MacBook Pro (Retina, 15-inch, Mid 2014) 2.2GHz quad-core Intel Core i7 @ 16GB RAM on SSD over an average of trials where feasible.</sup>

## Pre-converted Magnitude Formats of Popular Embeddings Models

Popular embedding models have been pre-converted to the `.magnitude` format for immmediate download and usage:

| **Contributor**                                                         | **Model**                                                       | **Light**<br/><br/><sup>(basic support for out-of-vocabulary keys)</sup>                                                                                                                                                                                                                                                                        | **Medium**<br/><i>(recommended)</i><br/><br/><sup>(advanced support for out-of-vocabulary keys)</sup>                                                                                                                                                                                                                                                                           | **Heavy**<br/><br/><sup>(advanced support for out-of-vocabulary keys and faster `most_similar_approx`)</sup>                                                                                                                                                                                                                                                                |
| :--------------------------------------------------------------------:  | :-------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Google - [word2vec](https://code.google.com/archive/p/word2vec/)        | Google News 100B                                                | [300D](http://magnitude.plasticity.ai/word2vec/GoogleNews-vectors-negative300.magnitude)                                                                                                                                                                                                                                                        | [300D](http://magnitude.plasticity.ai/word2vec+subword/GoogleNews-vectors-negative300.magnitude)                                                                                                                                                                                                                                                                                | [300D](http://magnitude.plasticity.ai/word2vec+approx/GoogleNews-vectors-negative300.magnitude)                                                                                                                                                                                                                                                                             |
| Stanford - [GloVE](https://nlp.stanford.edu/projects/glove/)            | Wikipedia 2014 + Gigaword 5 6B                                  | [50D](http://magnitude.plasticity.ai/glove/glove.6B.50d.magnitude),&nbsp;[100D](http://magnitude.plasticity.ai/glove/glove.6B.100d.magnitude),&nbsp;[200D](http://magnitude.plasticity.ai/glove/glove.6B.200d.magnitude),&nbsp;[300D](http://magnitude.plasticity.ai/glove/glove.6B.300d.magnitude)                                             | [50D](http://magnitude.plasticity.ai/glove+subword/glove.6B.50d.magnitude),&nbsp;[100D](http://magnitude.plasticity.ai/glove+subword/glove.6B.100d.magnitude),&nbsp;[200D](http://magnitude.plasticity.ai/glove+subword/glove.6B.200d.magnitude),&nbsp;[300D](http://magnitude.plasticity.ai/glove+subword/glove.6B.300d.magnitude)                                             | [50D](http://magnitude.plasticity.ai/glove+approx/glove.6B.50d.magnitude),&nbsp;[100D](http://magnitude.plasticity.ai/glove+approx/glove.6B.100d.magnitude),&nbsp;[200D](http://magnitude.plasticity.ai/glove+approx/glove.6B.200d.magnitude),&nbsp;[300D](http://magnitude.plasticity.ai/glove+approx/glove.6B.300d.magnitude)                                             |
| Stanford - [GloVE](https://nlp.stanford.edu/projects/glove/)            | Wikipedia 2014 + Gigaword 5 6B <br />(lemmatized by Plasticity) | [50D](http://magnitude.plasticity.ai/glove/glove-lemmatized.6B.50d.magnitude),&nbsp;[100D](http://magnitude.plasticity.ai/glove/glove-lemmatized.6B.100d.magnitude),&nbsp;[200D](http://magnitude.plasticity.ai/glove/glove-lemmatized.6B.200d.magnitude),&nbsp;[300D](http://magnitude.plasticity.ai/glove/glove-lemmatized.6B.300d.magnitude) | [50D](http://magnitude.plasticity.ai/glove+subword/glove-lemmatized.6B.50d.magnitude),&nbsp;[100D](http://magnitude.plasticity.ai/glove+subword/glove-lemmatized.6B.100d.magnitude),&nbsp;[200D](http://magnitude.plasticity.ai/glove+subword/glove-lemmatized.6B.200d.magnitude),&nbsp;[300D](http://magnitude.plasticity.ai/glove+subword/glove-lemmatized.6B.300d.magnitude) | [50D](http://magnitude.plasticity.ai/glove+approx/glove-lemmatized.6B.50d.magnitude),&nbsp;[100D](http://magnitude.plasticity.ai/glove+approx/glove-lemmatized.6B.100d.magnitude),&nbsp;[200D](http://magnitude.plasticity.ai/glove+approx/glove-lemmatized.6B.200d.magnitude),&nbsp;[300D](http://magnitude.plasticity.ai/glove+approx/glove-lemmatized.6B.300d.magnitude) |
| Stanford - [GloVE](https://nlp.stanford.edu/projects/glove/)            | Common Crawl 840B                                               | [300D](http://magnitude.plasticity.ai/glove/glove.840B.300d.magnitude)                                                                                                                                                                                                                                                                          | [300D](http://magnitude.plasticity.ai/glove+subword/glove.840B.300d.magnitude)                                                                                                                                                                                                                                                                                                  | [300D](http://magnitude.plasticity.ai/glove+approx/glove.840B.300d.magnitude)                                                                                                                                                                                                                                                                                               |
| Stanford - [GloVE](https://nlp.stanford.edu/projects/glove/)            | Twitter 27B                                                     | [25D](http://magnitude.plasticity.ai/glove/glove.twitter.27B.25d.magnitude),&nbsp;[50D](http://magnitude.plasticity.ai/glove/glove.twitter.27B.50d.magnitude),&nbsp;[100D](http://magnitude.plasticity.ai/glove/glove.twitter.27B.100d.magnitude),&nbsp;[200D](http://magnitude.plasticity.ai/glove/glove.twitter.27B.200d.magnitude)           | [25D](http://magnitude.plasticity.ai/glove+subword/glove.twitter.27B.25d.magnitude),&nbsp;[50D](http://magnitude.plasticity.ai/glove+subword/glove.twitter.27B.50d.magnitude),&nbsp;[100D](http://magnitude.plasticity.ai/glove+subword/glove.twitter.27B.100d.magnitude),&nbsp;[200D](http://magnitude.plasticity.ai/glove+subword/glove.twitter.27B.200d.magnitude)           | [25D](http://magnitude.plasticity.ai/glove+approx/glove.twitter.27B.25d.magnitude),&nbsp;[50D](http://magnitude.plasticity.ai/glove+approx/glove.twitter.27B.50d.magnitude),&nbsp;[100D](http://magnitude.plasticity.ai/glove+approx/glove.twitter.27B.100d.magnitude),&nbsp;[200D](http://magnitude.plasticity.ai/glove+approx/glove.twitter.27B.200d.magnitude)           |
| Facebook - [fastText](https://fasttext.cc/docs/en/english-vectors.html) | English Wikipedia 2017 16B                                      | [300D](http://magnitude.plasticity.ai/fasttext/wiki-news-300d-1M.magnitude)                                                                                                                                                                                                                                                                     | [300D](http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M.magnitude)                                                                                                                                                                                                                                                                                             | [300D](http://magnitude.plasticity.ai/fasttext+approx/wiki-news-300d-1M.magnitude)                                                                                                                                                                                                                                                                                          |
| Facebook - [fastText](https://fasttext.cc/docs/en/english-vectors.html) | English Wikipedia 2017 + subword 16B                            | [300D](http://magnitude.plasticity.ai/fasttext/wiki-news-300d-1M-subword.magnitude)                                                                                                                                                                                                                                                             | [300D](http://magnitude.plasticity.ai/fasttext+subword/wiki-news-300d-1M-subword.magnitude)                                                                                                                                                                                                                                                                                     | [300D](http://magnitude.plasticity.ai/fasttext+approx/wiki-news-300d-1M-subword.magnitude)                                                                                                                                                                                                                                                                                  |
| Facebook - [fastText](https://fasttext.cc/docs/en/english-vectors.html) | Common Crawl 600B                                               | [300D](http://magnitude.plasticity.ai/fasttext/crawl-300d-2M.magnitude)                                                                                                                                                                                                                                                                         | [300D](http://magnitude.plasticity.ai/fasttext+subword/crawl-300d-2M.magnitude)                                                                                                                                                                                                                                                                                                 | [300D](http://magnitude.plasticity.ai/fasttext+approx/crawl-300d-2M.magnitude)                                                                                                                                                                                                                                                                                              |

There are instructions [below](#file-format-and-converter) for converting any `.bin`, `.txt`, `.vec` file to a `.magnitude` file.

## Using the Library

#### Constructing a Magnitude Object

You can create a Magnitude object like so:
```python
from pymagnitude import *
vectors = Magnitude("/path/to/vectors.magnitude")
```

If needed, and included for convenience, you can also open a `.bin`, `.txt`, `.vec` file directly with Magnitude. This is, however, less efficient and very slow for large models as it will convert the file to a `.magnitude` file on the first run into a temporary directory. The temporary directory is not guaranteed to persist and does not persist when your computer reboots. You should [pre-convert `.bin`, `.txt`, `.vec` files with `python -m pymagnitude.converter`](#file-format-and-converter) typically for faster speeds, but this feature is useful for one-off use-cases. A warning will be generated when instantiating a Magnitude object directly with a `.bin`, `.txt`, `.vec`. You can supress warnings by setting the  `supress_warnings` argument in the constructor to `True`.

---------------

* <sup>By default, lazy loading is enabled. You can pass in an optional `lazy_loading` argument to the constructor with the value `-1` to disable lazy-loading and pre-load all vectors into memory (a la Gensim), `0` (default) to enable lazy-loading with an unbounded in-memory LRU cache, or an integer greater than zero `X` to enable lazy-loading with an LRU cache that holds the `X` most recently used vectors in memory.</sup> 
* <sup>If you want the data for the `most_similar` functions to be pre-loaded eagerly on initialization, set `eager` to `True`.</sup>
* <sup>Note, even when `lazy_loading` is set to `-1` or `eager` is set to `True` data will be pre-loaded into memory in a background thread to prevent the constructor from blocking for a few minutes for large models. If you really want blocking behavior, you can pass `True` to the `blocking` argument.</sup>
* <sup>By default, NumPy arrays are returned for queries. Set the optional argument `use_numpy` to `False` if you wish to recieve Python lists instead.</sup>
* <sup>By default, querying for keys is case-insensitive. Set the optional argument `case_insensitive` to `False` if you wish to perform case sensitive searches.</sup>
* <sup>Optionally, you can include the `pad_to_length` argument which will specify the length all examples should be padded to if passing in multple examples. Any examples that are longer than the pad length will be truncated.</sup>
* <sup>Optionally, you can set the `truncate_left` argument to `True` if you want the beginning of the the list of keys in each example to be truncated instead of the end in case it is longer than `pad_to_length` when specified.</sup>
* <sup>Optionally, you can set the `pad_left` argument to `True` if you want the padding to appear at the beginning versus the end (which is the default).</sup>
* <sup>Optionally, you can pass in the `placeholders` argument, which will increase the dimensions of each vector by a `placeholders` amount, zero-padding those extra dimensions. This is useful, if you plan to add other values and information to the vectors and want the space for that pre-allocated in the vectors for efficiency.</sup>

#### Querying

You can query the total number of vectors in the file like so:
```python
len(vectors)
```

---------------

You can query the dimensions of the vectors like so: 
```python
vectors.dim
```

---------------

You can check if a key is in the vocabulary like so: 
```python
"cat" in vectors
```

---------------

You can iterate through all keys and vectors like so:
```python
for key, vector in vectors:
  ...
```

---------------

You can query for the vector of a key like so: 
```python
vectors.query("cat")
```

---------------

You can index for the n-th key and vector like so:
```python
vectors[42]
```

---------------

You can query for the vector of multiple keys like so: 
```python
vectors.query(["I", "read", "a", "book"])
```
A 2D array (keys by vectors) will be returned.

---------------

You can query for the vector of multiple examples like so: 
```python
vectors.query([["I", "read", "a", "book"], ["I", "read", "a", "magazine"]])
```
A 3D array (examples by keys by vectors) will be returned. If `pad_to_length` is not specified, and the size of each example is uneven, they will be padded to the length of the longest example.

---------------

You can index for the keys and vectors of multiple indices like so:
```python
vectors[:42] # slice notation
vectors[42, 1337, 2001] # tuple notation
```

---------------

You can query the distance of two or multiple keys like so:
```python
vectors.distance("cat", "dog")
vectors.distance("cat", ["dog", "tiger"])
```

---------------

You can query the similarity of two or multiple keys like so:
```python
vectors.similarity("cat", "dog")
vectors.similarity("cat", ["dog", "tiger"])
```

---------------

You can query for the most similar key out of a list of keys to a given key like so:
```python
vectors.most_similar_to_given("cat", ["dog", "television", "laptop"]) # dog
```

---------------

You can query for which key doesn't match a list of keys to a given key like so:
```python
vectors.doesnt_match(["breakfast", "cereal", "dinner", "lunch"]) # cereal
```

---------------

You can query for the most similar (nearest neighbors) keys like so: 
```python
vectors.most_similar("cat", topn = 100) # Most similar by key
vectors.most_similar(vectors.query("cat"), topn = 100) # Most similar by vector
```
Optionally, you can pass a `max_distance` argument to `most_similar`. Since they are unit norm vectors, values from [0.0-2.0] are valid.

---------------

You can also query for the most similar keys giving positive and negative examples (which, incidentally, solves analogies) like so: 
```python
vectors.most_similar(positive = ["woman", "king"], negative = ["man"]) # queen
```

---------------

Similar to `vectors.most_similar`, a `vectors.most_similar_cosmul` function exists that uses the 3CosMul function from [Levy and Goldberg](http://www.aclweb.org/anthology/W14-1618):
```python
vectors.most_similar_cosmul(positive = ["woman", "king"], negative = ["man"]) # queen
```

---------------

You can also query for the most similar keys using an approximate nearest neighbors index which is much faster, but doesn't guarantee the exact answer: 
```python
vectors.most_similar_approx("cat")
vectors.most_similar_approx(positive = ["woman", "king"], negative = ["man"])
```
Optionally, you can pass an `effort` argument with values between [0.0-1.0] to the `most_similar_approx` function which will give you runtime trade-off. The default value for `effort` is 1.0 which will take the longest, but will give the most accurate result.

---------------

You can query for all keys closer to a key than another key is like so:
```python
vectors.closer_than("cat", "rabbit") # ["dog", ...]
```

---------------

You can access all of the underlying vectors in the model in a large `numpy.memmap` array of size (`len(vectors) x vectors.emb_dim`) like so:

```python
vectors.get_vectors_mmap()
```

---------------

You can clean up all associated resources, open files, and database connections like so:
```python
vectors.close()
```

### Basic Out-of-Vocabulary Keys

For word vector representations, handling out-of-vocabulary keys is important to handling new words not in the trained model, handling mispellings and typos, and making models trained on the word vector representations more robust in general.

Out-of-vocabulary keys are handled by assigning them a random vector value. However, the randomness is deterministic. So if the *same* out-of-vocabulary key is encountered twice, it will be assigned the same random vector value for the sake of being able to train on those out-of-vocabulary keys. Moreover, if two out-of-vocabulary keys share similar character n-grams ("uberx", "uberxl") they will placed close to each other even if they are both not in the vocabulary:

```python
vectors = Magnitude("/path/to/GoogleNews-vectors-negative300.magnitude")
"uberx" in vectors # False
"uberxl" in vectors # False
vectors.query("uberx") # array([ 5.07109939e-02, -7.08248823e-02, -2.74812328e-02, ... ])
vectors.query("uberxl") # array([ 0.04734962, -0.08237578, -0.0333479, -0.00229564, ... ])
vectors.similarity("uberx", "uberxl") # 0.955000000200815
```

### Advanced Out-of-Vocabulary Keys

If using a Magnitude file with advanced out-of-vocabulary support (Medium or Heavy), out-of-vocabulary keys will also be embedded close to similar keys (determined by string similarity) that *are in* the vocabulary:
```python
vectors = Magnitude("/path/to/GoogleNews-vectors-negative300.magnitude")
"uberx" in vectors # False
"uber" in vectors # True
vectors.similarity("uberx", "uber") # 0.7383483267618451
```

#### Misspellings
This also makes Magnitude robust to a lot of spelling errors:
```python
vectors = Magnitude("/path/to/GoogleNews-vectors-negative300.magnitude")
"missispi" in vectors # False
vectors.similarity("missispi", "mississippi") # 0.35961736624824003
"discrimnatory" in vectors # False
vectors.similarity("discrimnatory", "discriminatory") # 0.8309152561753461
"hiiiiiiiiii" in vectors # False
vectors.similarity("hiiiiiiiiii", "hi") # 0.7069775034853861
```

Character n-grams are used to create this effect for out-of-vocabulary keys. The inspiration for this feature was taken from Facebook AI Research's [Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606.pdf), but instead of utilizing character n-grams at train time, character n-grams are used at inference so the effect can be somewhat replicated (but not perfectly replicated) in older models that were not trained with character n-grams like word2vec and GloVE.

### Concatenation
Optionally, you can combine vectors from multiple models to feed stronger information into a machine learning model like so:
```python
from pymagnitude import *
word2vec = Magnitude("/path/to/GoogleNews-vectors-negative300.magnitude")
glove = Magnitude("/path/to/glove.6B.50d.magnitude")
vectors = Magnitude(word2vec, glove) # concatenate word2vec with glove
vectors.query("cat") # returns 350-dimensional NumPy array ('cat' from word2vec concatenated with 'cat' from glove)
vectors.query(("cat", "cats")) # returns 350-dimensional NumPy array ('cat' from word2vec concatenated with 'cats' from glove)
```

You can concatenate more than two vector models, simply by passing more arguments to constructor.

### Additional Featurization (Parts of Speech, etc.)


## Concurrency and Parallelism
The library is thread safe (it uses a different connection to the underlying store per thread), is read-only, and it never writes to the file. Because of the light-memory usage, you can also run it in multiple processes (or use `multiprocessing`) with different address spaces without having to duplicate the data in-memory like with other libraries like Gensim and without having to create a multi-process shared variable since data is read off-disk and each process keeps its own LRU memory cache. For heavier functions, like `most_similar` a shared memory mapped file is created to share memory between processes.

## File Format and Converter
The Magnitude package uses the `.magnitude` file format instead of `.bin`, `.txt`, or `.vec` as with other vector models like word2vec, GloVE, and fastText. There is an included command-line utility for converting word2vec, GloVE, fastText files to Magnitude files.

You can convert them like so:
```bash
python -m pymagnitude.converter -i <PATH TO FILE TO BE CONVERTED> -o <OUTPUT PATH FOR MAGNITUDE FILE>
```

The input format will automatically be determined by the extension / the contents of the input file. When the vectors are converted, they will also be [unit-length normalized](https://en.wikipedia.org/wiki/Unit_vector). You should only need to perform this conversion once for a model. After converting, the Magnitude file format is static and it will not be modified or written to make concurrent read access safe.

The flags for  `pymagnitude.converter` are specified below:
* You can pass in the `-h` flag for help and to list all flags.
* You can use the `-p <PRECISION>` flag to specify the decimal precision to retain (selecting a lower number will create smaller files). The actual underlying values are stored as integers instead of floats so this is essentially [quantization](https://www.tensorflow.org/performance/quantization) for smaller model footprints.
* You can add an approximate nearest neighbors index to the file (increases size) with the `-a` flag which will enable the use of the `most_similar_approx` function. The `-t <TREES>` flag controls the number of trees in the approximate neigherest neighbors index (higher is more accurcate) when used in conjunction with the `-a` flag (if not supplied, the number of trees is automatically determined).
* You can pass the `-s` flag to disable adding subword information to the file (which will make the file smaller), but disable advanced out-of-vocabulary key support.

Optionally, you can bulk convert many files by passing an input folder and output folder instead of an input file and output file. All `.txt`, `.bin`, and `.vec` files in the input folder will be converted to `.magnitude` files in the the output folder. The output folder must exist before a bulk conversion operation.

## Other Documentation
Other documentation is not available at this time. See the source file directly (it is well commented) if you need more information about a method's arguments or want to see all supported features.

## Other Languages
Currently, reading Magnitude files is only supported in Python, since it has become the de-facto language for machine learning. This is sufficient for most use cases. Extending the file format to other languages shouldn't be difficult as SQLite has a native C implementation and has bindings in most languages. The file format itself and the protocol for reading and searching is also fairly straightforward upon reading the source code of this repository.

## Other Domains
Currently, natural language processing is the most popular domain that uses pre-trained vector embedding models for word vector representations. There are, however, other domains like computer vision that have started using pre-trained vector embedding models like [Deep1B](https://github.com/arbabenko/GNOIMI) for image representation. This library intends to stay agnostic to various domains and instead provides a generic key-vector store and interface that is useful for all domains.

## Contributing
The main repository for this project can be found on [GitLab](https://gitlab.com/Plasticity/magnitude). The [GitHub repository](https://github.com/plasticityai/magnitude) is only a mirror. Pull requests for more tests, better error-checking, bug fixes, performance improvements, or adding additional utilties / functionalities are welcome on [GitLab](https://gitlab.com/Plasticity/magnitude).

You can contact us at [opensource@plasticity.ai](mailto:opensource@plasticity.ai).

## Other Notable Projects
* [spotify/annoy](https://github.com/spotify/annoy) - Powers the approximate nearest neighbors algorithm behind `most_similar_approx` in Magnitude using random-projection trees and hierarchical 2-means. Thanks to author [Erik Bernhardsson](https://github.com/erikbern) for helping out with some of the integration details between Magnitude and Annoy.

## LICENSE and Attribution

This repository is licensed under the license found [here](https://gitlab.com/Plasticity/magnitude/blob/master/LICENSE.txt).

“[Seismic](https://thenounproject.com/ziman.jan/collection/weather/?i=1518266)” icon by JohnnyZi from the [Noun Project](https://thenounproject.com).
