u"""
Various utilities that don't fit anwhere else.
"""



from __future__ import division
from __future__ import absolute_import
try:
    from itertools import zip_longest, islice
except:
    from itertools import izip_longest
#typing
import importlib
import logging
import pkgutil
import random
import subprocess
import sys
import os

try:
    import resource
except ImportError:
    # resource doesn't exist on Windows systems
    resource = None

import torch
import numpy
import spacy
from spacy.cli.download import download as spacy_download
from spacy.language import Language as SpacyModelType

# This base import is so we can refer to allennlp.data.Token in `sanitize()` without creating
# circular dependencies.
import allennlp
from allennlp.common.checks import log_pytorch_version_info
from allennlp.common.params import Params
from allennlp.common.tqdm import Tqdm
from allennlp.common.tee_logger import TeeLogger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

JsonDict = {}  # pylint: disable=invalid-name

# If you want to have start and/or end symbols for any reason in your code, we recommend you use
# these, to have a common place to import from.  Also, it's important for some edge cases in how
# data is processed for these symbols to be lowercase, not uppercase (because we have code that
# will lowercase tokens for you in some circumstances, and we need this symbol to not change in
# those cases).
START_SYMBOL = u'@start@'
END_SYMBOL = u'@end@'


def sanitize(x     )       :  # pylint: disable=invalid-name,too-many-return-statements
    u"""
    Sanitize turns PyTorch and Numpy types into basic Python types so they
    can be serialized into JSON.
    """
    if isinstance(x, (unicode, float, int, bool)):
        # x is already serializable
        return x
    elif isinstance(x, torch.Tensor):
        # tensor needs to be converted to a list (and moved to cpu if necessary)
        return x.cpu().tolist()
    elif isinstance(x, numpy.ndarray):
        # array needs to be converted to a list
        return x.tolist()
    elif isinstance(x, numpy.number):
        # NumPy numbers need to be converted to Python numbers
        return x.item()
    elif isinstance(x, dict):
        # Dicts need their values sanitized
        return dict((key, sanitize(value)) for key, value in list(x.items()))
    elif isinstance(x, (list, tuple)):
        # Lists and Tuples need their values sanitized
        return [sanitize(x_i) for x_i in x]
    elif isinstance(x, (spacy.tokens.Token, allennlp.data.Token)):
        # Tokens get sanitized to just their text.
        return x.text
    elif x is None:
        return u"None"
    elif hasattr(x, u'to_json'):
        return x.to_json()
    else:
        raise ValueError("Cannot sanitize {x} of type {type(x)}. "
                         u"If this is your own custom class, add a `to_json(self)` method "
                         u"that returns a JSON-like object.")

def group_by_count(iterable           , count     , default_value     )                   :
    u"""
    Takes a list and groups it into sublists of size ``count``, using ``default_value`` to pad the
    list at the end if the list is not divisable by ``count``.

    For example:
    >>> group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 0)
    [[1, 2, 3], [4, 5, 6], [7, 0, 0]]

    This is a short method, but it's complicated and hard to remember as a one-liner, so we just
    make a function out of it.
    """
    return [list(l) for l in zip_longest(*[iter(iterable)] * count, fillvalue=default_value)]



def lazy_groups_of(iterator             , group_size     )                     :
    u"""
    Takes an iterator and batches the invididual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    return iter(lambda: list(islice(iterator, 0, group_size)), [])

def pad_sequence_to_length(sequence      ,
                           desired_length     ,
                           default_value                    = lambda: 0,
                           padding_on_right       = True)        :
    u"""
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.

    Parameters
    ----------
    sequence : List
        A list of objects to be padded.

    desired_length : int
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.

    default_value: Callable, default=lambda: 0
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.

    padding_on_right : bool, default=True
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?

    Returns
    -------
    padded_sequence : List
    """
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    for _ in range(desired_length - len(padded_sequence)):
        if padding_on_right:
            padded_sequence.append(default_value())
        else:
            padded_sequence.insert(0, default_value())
    return padded_sequence


def add_noise_to_dict_values(dictionary                , noise_param       )                  :
    u"""
    Returns a new dictionary with noise added to every key in ``dictionary``.  The noise is
    uniformly distributed within ``noise_param`` percent of the value for every value in the
    dictionary.
    """
    new_dict = {}
    for key, value in list(dictionary.items()):
        noise_value = value * noise_param
        noise = random.uniform(-noise_value, noise_value)
        new_dict[key] = value + noise
    return new_dict


def namespace_match(pattern     , namespace     ):
    u"""
    Matches a namespace pattern against a namespace string.  For example, ``*tags`` matches
    ``passage_tags`` and ``question_tags`` and ``tokens`` matches ``tokens`` but not
    ``stemmed_tokens``.
    """
    if pattern[0] == u'*' and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False


def prepare_environment(params        ):
    u"""
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.

    Parameters
    ----------
    params: Params object or dict, required.
        A ``Params`` object or dict holding the json parameters.
    """
    seed = params.pop_int(u"random_seed", 13370)
    numpy_seed = params.pop_int(u"numpy_seed", 1337)
    torch_seed = params.pop_int(u"pytorch_seed", 133)

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        numpy.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        # Seed all GPUs with the same seed if available.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)

    log_pytorch_version_info()

def prepare_global_logging(serialization_dir     , file_friendly_logging      )        :
    u"""
    This function configures 3 global logging attributes - streaming stdout and stderr
    to a file as well as the terminal, setting the formatting for the python logging
    library and setting the interval frequency for the Tqdm progress bar.

    Note that this function does not set the logging level, which is set in ``allennlp/run.py``.

    Parameters
    ----------
    serializezation_dir : ``str``, required.
        The directory to stream logs to.
    file_friendly_logging : ``bool``, required.
        Whether logs should clean the output to prevent carridge returns
        (used to update progress bars on a single terminal line).
    """
    Tqdm.set_slower_interval(file_friendly_logging)
    std_out_file = os.path.join(serialization_dir, u"stdout.log")
    sys.stdout = TeeLogger(std_out_file, # type: ignore
                           sys.stdout,
                           file_friendly_logging)
    sys.stderr = TeeLogger(os.path.join(serialization_dir, u"stderr.log"), # type: ignore
                           sys.stderr,
                           file_friendly_logging)

    stdout_handler = logging.FileHandler(std_out_file)
    stdout_handler.setFormatter(logging.Formatter(u'%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger().addHandler(stdout_handler)

LOADED_SPACY_MODELS                                                     = {}


def get_spacy_model(spacy_model_name     , pos_tags      , parse      , ner      )                  :
    u"""
    In order to avoid loading spacy models a whole bunch of times, we'll save references to them,
    keyed by the options we used to create the spacy model, so any particular configuration only
    gets loaded once.
    """

    options = (spacy_model_name, pos_tags, parse, ner)
    if options not in LOADED_SPACY_MODELS:
        disable = [u'vectors', u'textcat']
        if not pos_tags:
            disable.append(u'tagger')
        if not parse:
            disable.append(u'parser')
        if not ner:
            disable.append(u'ner')
        try:
            spacy_model = spacy.load(spacy_model_name, disable=disable)
        except OSError:
            logger.warning("Spacy models '{spacy_model_name}' not found.  Downloading and installing.")
            spacy_download(spacy_model_name)
            spacy_model = spacy.load(spacy_model_name, disable=disable)

        LOADED_SPACY_MODELS[options] = spacy_model
    return LOADED_SPACY_MODELS[options]

def import_submodules(package_name     )        :
    u"""
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    importlib.invalidate_caches()

    # Import at top level
    module = importlib.import_module(package_name)
    path = getattr(module, u'__path__', [])

    # walk_packages only finds immediate children, so need to recurse.
    for _, name, _ in pkgutil.walk_packages(path):
        subpackage = "{package_name}.{name}"
        import_submodules(subpackage)


def peak_memory_mb()         :
    u"""
    Get peak memory usage for this process, as measured by
    max-resident-set size:

    https://unix.stackexchange.com/questions/30940/getrusage-system-call-what-is-maximum-resident-set-size

    Only works on OSX and Linux, returns 0.0 otherwise.
    """
    if resource is None or sys.platform not in (u'linux', u'darwin'):
        return 0.0

    # TODO(joelgrus): For whatever, our pinned version 0.521 of mypy does not like
    # next line, but later versions (e.g. 0.530) are fine with it. Once we get that
    # figured out, remove the type: ignore.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore

    if sys.platform == u'darwin':
        # On OSX the result is in bytes.
        return peak / 1000000

    else:
        # On Linux the result is in kilobytes.
        return peak / 1000

def gpu_memory_mb()                  :
    u"""
    Get the current GPU memory usage.
    Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Returns
    -------
    ``Dict[int, int]``
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
        Returns an empty ``dict`` if GPUs are not available.
    """
    # pylint: disable=bare-except
    try:
        result = subprocess.check_output([u'nvidia-smi', u'--query-gpu=memory.used',
                                          u'--format=csv,nounits,noheader'],
                                         encoding=u'utf-8')
        gpu_memory = [int(x) for x in result.strip().split(u'\n')]
        return dict((gpu, memory) for gpu, memory in enumerate(gpu_memory))
    except FileNotFoundError:
        # `nvidia-smi` doesn't exist, assume that means no GPU.
        return {}
    except:
        # Catch *all* exceptions, because this memory check is a nice-to-have
        # and we'd never want a training run to fail because of it.
        logger.exception(u"unable to check gpu_memory_mb(), continuing")
        return {}


def ensure_list(iterable             )           :
    u"""
    An Iterable may be a list or a generator.
    This ensures we get a list without making an unnecessary copy.
    """
    if isinstance(iterable, list):
        return iterable
    else:
        return list(iterable)

def is_lazy(iterable             )        :
    u"""
    Checks if the given iterable is lazy,
    which here just means it's not a list.
    """
    return not isinstance(iterable, list)

def get_frozen_and_tunable_parameter_names(model                 )        :
    frozen_parameter_names = []
    tunable_parameter_names = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            frozen_parameter_names.append(name)
        else:
            tunable_parameter_names.append(name)
    return [frozen_parameter_names, tunable_parameter_names]
