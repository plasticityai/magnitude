# pylint: disable=no-self-use

from __future__ import absolute_import
#typing

#overrides

from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.common.util import pad_sequence_to_length


class ListField(SequenceField):
    u"""
    A ``ListField`` is a list of other fields.  You would use this to represent, e.g., a list of
    answer options that are themselves ``TextFields``.

    This field will get converted into a tensor that has one more mode than the items in the list.
    If this is a list of ``TextFields`` that have shape (num_words, num_characters), this
    ``ListField`` will output a tensor of shape (num_sentences, num_words, num_characters).

    Parameters
    ----------
    field_list : ``List[Field]``
        A list of ``Field`` objects to be concatenated into a single input tensor.  All of the
        contained ``Field`` objects must be of the same type.
    """
    def __init__(self, field_list             )        :
        field_class_set = set([field.__class__ for field in field_list])
        assert len(field_class_set) == 1, u"ListFields must contain a single field type, found " +\
                                          unicode(field_class_set)
        # Not sure why mypy has a hard time with this type...
        self.field_list = field_list

    #overrides
    def count_vocab_items(self, counter                           ):
        for field in self.field_list:
            field.count_vocab_items(counter)

    #overrides
    def index(self, vocab            ):
        for field in self.field_list:
            field.index(vocab)

    #overrides
    def get_padding_lengths(self)                  :
        field_lengths = [field.get_padding_lengths() for field in self.field_list]
        padding_lengths = {u'num_fields': len(self.field_list)}

        # We take the set of all possible padding keys for all fields, rather
        # than just a random key, because it is possible for fields to be empty
        # when we pad ListFields.
        possible_padding_keys = [key for field_length in field_lengths
                                 for key in list(field_length.keys())]

        for key in set(possible_padding_keys):
            # In order to be able to nest ListFields, we need to scope the padding length keys
            # appropriately, so that nested ListFields don't all use the same "num_fields" key.  So
            # when we construct the dictionary from the list of fields, we add something to the
            # name, and we remove it when padding the list of fields.
            padding_lengths[u'list_' + key] = max(x[key] if key in x else 0 for x in field_lengths)
        return padding_lengths

    #overrides
    def sequence_length(self)       :
        return len(self.field_list)

    #overrides
    def as_tensor(self,
                  padding_lengths                ,
                  cuda_device      = -1)             :
        padded_field_list = pad_sequence_to_length(self.field_list,
                                                   padding_lengths[u'num_fields'],
                                                   self.field_list[0].empty_field)
        # Here we're removing the scoping on the padding length keys that we added in
        # `get_padding_lengths`; see the note there for more detail.
        child_padding_lengths = dict((key.replace(u'list_', u'', 1), value)
                                 for key, value in list(padding_lengths.items())
                                 if key.startswith(u'list_'))
        padded_fields = [field.as_tensor(child_padding_lengths, cuda_device)
                         for field in padded_field_list]
        return self.field_list[0].batch_tensors(padded_fields)

    #overrides
    def empty_field(self):
        # Our "empty" list field will actually have a single field in the list, so that we can
        # correctly construct nested lists.  For example, if we have a type that is
        # `ListField[ListField[LabelField]]`, we need the top-level `ListField` to know to
        # construct a `ListField[LabelField]` when it's padding, and the nested `ListField` needs
        # to know that it's empty objects are `LabelFields`.  Having an "empty" list actually have
        # length one makes this all work out, and we'll always be padding to at least length 1,
        # anyway.
        return ListField([self.field_list[0].empty_field()])

    #overrides
    def batch_tensors(self, tensor_list                 )             :
        # We defer to the class we're wrapping in a list to handle the batching.
        return self.field_list[0].batch_tensors(tensor_list)

    def __str__(self)       :
        field_class = self.field_list[0].__class__.__name__
        base_string = "ListField of {len(self.field_list)} {field_class}s : \n"
        return u" ".join([base_string] + ["\t {field} \n" for field in self.field_list])
