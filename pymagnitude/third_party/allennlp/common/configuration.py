u"""
Tools for programmatically generating config files for AllenNLP models.
"""
# pylint: disable=protected-access,too-many-return-statements


from __future__ import absolute_import
from __future__ import print_function
#typing
import collections
import inspect
import importlib
import json
import re

import torch
from numpydoc.docscrape import NumpyDocString

from allennlp.common import Registrable, JsonDict
from allennlp.data.iterators import DataIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules.seq2seq_encoders import _Seq2SeqWrapper
from allennlp.modules.seq2vec_encoders import _Seq2VecWrapper
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.initializers import Initializer
from allennlp.nn.regularizers import Regularizer
from allennlp.training.optimizers import Optimizer as AllenNLPOptimizer
from allennlp.training.trainer import Trainer
try:
    from itertools import izip
except:
    izip = zip


def _remove_prefix(class_name     )       :
    rgx = r"^(typing\.|builtins\.)"
    return re.sub(rgx, u"", class_name)

def full_name(cla55                )       :
    u"""
    Return the full name (including module) of the given class.
    """
    # Special case to handle None:
    if cla55 is None:
        return u"?"

    if issubclass(cla55, Initializer) and cla55 != Initializer:
        init_fn = cla55()._init_function
        return "{init_fn.__module__}.{init_fn.__name__}"

    origin = getattr(cla55, u'__origin__', None)
    args = getattr(cla55, u'__args__', ())

    # Special handling for compound types
    if origin in (Dict, dict):
        key_type, value_type = args
        return """Dict[{full_name(key_type)}, {full_name(value_type)}]"""
    elif origin in (Tuple, tuple, List, list, Sequence, collections.abc.Sequence):
        return """{_remove_prefix(str(origin))}[{", ".join(full_name(arg) for arg in args)}]"""
    elif origin == Union:
        # Special special case to handle optional types:
        if len(args) == 2 and args[-1] == type(None):
            return """Optional[{full_name(args[0])}]"""
        else:
            return """Union[{", ".join(full_name(arg) for arg in args)}]"""
    else:
        return _remove_prefix("{cla55.__module__}.{cla55.__name__}")


def json_annotation(cla55                ):
    # Special case to handle None:
    if cla55 is None:
        return {u'origin': u'?'}

    # Hack because e.g. typing.Union isn't a type.
    if isinstance(cla55, type) and issubclass(cla55, Initializer) and cla55 != Initializer:
        init_fn = cla55()._init_function
        return {u'origin': "{init_fn.__module__}.{init_fn.__name__}"}

    origin = getattr(cla55, u'__origin__', None)
    args = getattr(cla55, u'__args__', ())

    # Special handling for compound types
    if origin in (Dict, dict):
        key_type, value_type = args
        return {u'origin': u"Dict", u'args': [json_annotation(key_type), json_annotation(value_type)]}
    elif origin in (Tuple, tuple, List, list, Sequence, collections.abc.Sequence):
        return {u'origin': _remove_prefix(unicode(origin)), u'args': [json_annotation(arg) for arg in args]}
    elif origin == Union:
        # Special special case to handle optional types:
        if len(args) == 2 and args[-1] == type(None):
            return json_annotation(args[0])
        else:
            return {u'origin': u"Union", u'args': [json_annotation(arg) for arg in args]}
    elif cla55 == Ellipsis:
        return {u'origin': u"..."}
    else:
        return {u'origin': _remove_prefix("{cla55.__module__}.{cla55.__name__}")}


class ConfigItem():
    u"""
    Each ``ConfigItem`` represents a single entry in a configuration JsonDict.
    """
    #ame: str
    #nnotation: type
    default_value                = None
    comment      = u''

    def to_json(self)            :
        json_dict = {
                u"name": self.name,
                u"annotation": json_annotation(self.annotation),
        }

        if is_configurable(self.annotation):
            json_dict[u"configurable"] = True

        if self.default_value != _NO_DEFAULT:
            try:
                # Ugly check that default value is actually serializable
                json.dumps(self.default_value)
                json_dict[u"defaultValue"] = self.default_value
            except TypeError:
                print("unable to json serialize {self.default_value}, using None instead")
                json_dict[u"defaultValue"] = None


        if self.comment:
            json_dict[u"comment"] = self.comment

        return json_dict


# typevar


class Config():
    u"""
    A ``Config`` represents an entire subdict in a configuration file.
    If it corresponds to a named subclass of a registrable class,
    it will also contain a ``type`` item in addition to whatever
    items are required by the subclass ``from_params`` method.
    """
    def __init__(self, items                  , typ3      = None)        :
        self.items = items
        self.typ3 = typ3

    def __repr__(self)       :
        return "Config({self.items})"

    def to_json(self)            :
        blob           = {u'items': [item.to_json() for item in self.items]}

        if self.typ3:
            blob[u"type"] = self.typ3

        return blob


# ``None`` is sometimes the default value for a function parameter,
# so we use a special sentinel to indicate that a parameter has no
# default value.
_NO_DEFAULT = object()

def _get_config_type(cla55      )                 :
    u"""
    Find the name (if any) that a subclass was registered under.
    We do this simply by iterating through the registry until we
    find it.
    """
    # Special handling for pytorch RNN types:
    if cla55 == torch.nn.RNN:
        return u"rnn"
    elif cla55 == torch.nn.LSTM:
        return u"lstm"
    elif cla55 == torch.nn.GRU:
        return u"gru"

    for subclass_dict in list(Registrable._registry.values()):
        for name, subclass in list(subclass_dict.items()):
            if subclass == cla55:
                return name

        # Special handling for initializer functions
            if hasattr(subclass, u'_initializer_wrapper'):
                sif = subclass()._init_function
                if sif == cla55:
                    return sif.__name__.rstrip(u"_")

    return None

def _docspec_comments(obj)                  :
    u"""
    Inspect the docstring and get the comments for each parameter.
    """
    # Sometimes our docstring is on the class, and sometimes it's on the initializer,
    # so we've got to check both.
    class_docstring = getattr(obj, u'__doc__', None)
    init_docstring = getattr(obj.__init__, u'__doc__', None) if hasattr(obj, u'__init__') else None

    docstring = class_docstring or init_docstring or u''

    doc = NumpyDocString(docstring)
    params = doc[u"Parameters"]
    comments                 = {}

    for line in params:
        # It looks like when there's not a space after the parameter name,
        # numpydocstring parses it incorrectly.
        name_bad = line[0]
        name = name_bad.split(u":")[0]

        # Sometimes the line has 3 fields, sometimes it has 4 fields.
        comment = u"\n".join(line[-1])

        comments[name] = comment

    return comments

def _auto_config(cla55         )             :
    u"""
    Create the ``Config`` for a class by reflecting on its ``__init__``
    method and applying a few hacks.
    """
    typ3 = _get_config_type(cla55)

    # Don't include self, or vocab
    names_to_ignore = set([u"self", u"vocab"])

    # Hack for RNNs
    if cla55 in [torch.nn.RNN, torch.nn.LSTM, torch.nn.GRU]:
        cla55 = torch.nn.RNNBase
        names_to_ignore.add(u"mode")

    if isinstance(cla55, type):
        # It's a class, so inspect its constructor
        function_to_inspect = cla55.__init__
    else:
        # It's a function, so inspect it, and ignore tensor
        function_to_inspect = cla55
        names_to_ignore.add(u"tensor")

    argspec = inspect.getargspec(function_to_inspect)
    comments = _docspec_comments(cla55)

    items                   = []

    num_args = len(argspec.args)
    defaults = list(argspec.defaults or [])
    num_default_args = len(defaults)
    num_non_default_args = num_args - num_default_args

    # Required args all come first, default args at the end.
    defaults = [_NO_DEFAULT for _ in range(num_non_default_args)] + defaults

    for name, default in izip(argspec.args, defaults):
        if name in names_to_ignore:
            continue
        annotation = argspec.annotations.get(name)
        comment = comments.get(name)

        # Don't include Model, the only place you'd specify that is top-level.
        if annotation == Model:
            continue

        # Don't include DataIterator, the only place you'd specify that is top-level.
        if annotation == DataIterator:
            continue

        # Don't include params for an Optimizer
        if torch.optim.Optimizer in getattr(cla55, u'__bases__', ()) and name == u"params":
            continue

        # Don't include datasets in the trainer
        if cla55 == Trainer and name.endswith(u"_dataset"):
            continue

        # Hack in our Optimizer class to the trainer
        if cla55 == Trainer and annotation == torch.optim.Optimizer:
            annotation = AllenNLPOptimizer

        # Hack in embedding num_embeddings as optional (it can be inferred from the pretrained file)
        if cla55 == Embedding and name == u"num_embeddings":
            default = None

        items.append(ConfigItem(name, annotation, default, comment))

    # More hacks, Embedding
    if cla55 == Embedding:
        items.insert(1, ConfigItem(u"pretrained_file", unicode, None))

    return Config(items, typ3=typ3)


def render_config(config        , indent      = u"")       :
    u"""
    Pretty-print a config in sort-of-JSON+comments.
    """
    # Add four spaces to the indent.
    new_indent = indent + u"    "

    return u"".join([
            # opening brace + newline
            u"{\n",
            # "type": "...", (if present)
            '{new_indent}"type": "{config.typ3}",\n' if config.typ3 else u'',
            # render each item
            u"".join(_render(item, new_indent) for item in config.items),
            # indent and close the brace
            indent,
            u"}\n"
    ])


def _remove_optional(typ3      )        :
    origin = getattr(typ3, u'__origin__', None)
    args = getattr(typ3, u'__args__', None)

    if origin == Union and len(args) == 2 and args[-1] == type(None):
        return _remove_optional(args[0])
    else:
        return typ3

def is_configurable(typ3      )        :
    # Throw out optional:
    typ3 = _remove_optional(typ3)

    # Anything with a from_params method is itself configurable.
    # So are regularizers even though they don't.
    return any([
            hasattr(typ3, u'from_params'),
            typ3 == Regularizer,
    ])

def _render(item            , indent      = u"")       :
    u"""
    Render a single config item, with the provided indent
    """
    optional = item.default_value != _NO_DEFAULT

    if is_configurable(item.annotation):
        rendered_annotation = "{item.annotation} (configurable)"
    else:
        rendered_annotation = unicode(item.annotation)

    rendered_item = u"".join([
            # rendered_comment,
            indent,
            u"// " if optional else u"",
            '"{item.name}": ',
            rendered_annotation,
            " (default: {item.default_value} )" if optional else u"",
            " // {item.comment}" if item.comment else u"",
            u"\n"
    ])

    return rendered_item

BASE_CONFIG         = None
def _valid_choices(cla55      )                  :
    u"""
    Return a mapping {registered_name -> subclass_name}
    for the registered subclasses of `cla55`.
    """
    choices                 = {}

    if cla55 not in Registrable._registry:
        raise ValueError("{cla55} is not a known Registrable class")

    for name, subclass in list(Registrable._registry[cla55].items()):
        # These wrapper classes need special treatment
        if isinstance(subclass, (_Seq2SeqWrapper, _Seq2VecWrapper)):
            subclass = subclass._module_class

        choices[name] = full_name(subclass)

    return choices

def configure(full_path      = u'')                            :
    if not full_path:
        return BASE_CONFIG

    parts = full_path.split(u".")
    class_name = parts[-1]
    module_name = u".".join(parts[:-1])
    module = importlib.import_module(module_name)
    cla55 = getattr(module, class_name)

    if Registrable in getattr(cla55, u'__bases__', ()):
        return list(_valid_choices(cla55).values())
    else:
        return _auto_config(cla55)
