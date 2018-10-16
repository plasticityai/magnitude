# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from __future__ import absolute_import
#typing

from allennlp.common import Params
from allennlp.common.from_params import FromParams, takes_arg, remove_optional, create_kwargs
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers.word_splitter import WordSplitter

class MyClass(FromParams):
    def __init__(self, my_int     , my_bool       = False)        :
        self.my_int = my_int
        self.my_bool = my_bool


class TestFromParams(AllenNlpTestCase):
    def test_takes_arg(self):
        def bare_function(some_input     )       :
            return some_input + 1

        assert takes_arg(bare_function, u'some_input')
        assert not takes_arg(bare_function, u'some_other_input')

        class SomeClass(object):
            total = 0

            def __init__(self, constructor_param     )        :
                self.constructor_param = constructor_param

            def check_param(self, check     )        :
                return self.constructor_param == check

            @classmethod
            def set_total(cls, new_total     )        :
                cls.total = new_total

        assert takes_arg(SomeClass, u'self')
        assert takes_arg(SomeClass, u'constructor_param')
        assert not takes_arg(SomeClass, u'check')

        assert takes_arg(SomeClass.check_param, u'check')
        assert not takes_arg(SomeClass.check_param, u'other_check')

        assert takes_arg(SomeClass.set_total, u'new_total')
        assert not takes_arg(SomeClass.set_total, u'total')

    def test_remove_optional(self):
        optional_type = Optional[Dict[unicode, unicode]]
        bare_type = remove_optional(optional_type)
        bare_bare_type = remove_optional(bare_type)

        assert bare_type == Dict[unicode, unicode]
        assert bare_bare_type == Dict[unicode, unicode]

        assert remove_optional(Optional[unicode]) == unicode
        assert remove_optional(unicode) == unicode

    def test_from_params(self):
        my_class = MyClass.from_params(Params({u"my_int": 10}), my_bool=True)

        assert isinstance(my_class, MyClass)
        assert my_class.my_int == 10
        assert my_class.my_bool

    def test_create_kwargs(self):
        kwargs = create_kwargs(MyClass,
                               Params({u'my_int': 5}),
                               my_bool=True,
                               my_float=4.4)

        # my_float should not be included because it's not a param of the MyClass constructor
        assert kwargs == {
                u"my_int": 5,
                u"my_bool": True
        }

    def test_extras(self):
        # pylint: disable=unused-variable,arguments-differ
        from allennlp.common.registrable import Registrable

        class A(Registrable):
            pass

        class B(A):
            def __init__(self, size     , name     )        :
                self.size = size
                self.name = name

        B = A.register(u"b")(B)

class C(A):
            def __init__(self, size     , name     )        :
                self.size = size
                self.name = name

            # custom from params
            @classmethod
            def from_params(cls, params        , size     )       :  # type: ignore
                name = params.pop(u'name')
                return cls(size=size, name=name)


        # Check that extras get passed, even though A doesn't need them.
        C = A.register(u"c")(C)

params = Params({u"type": u"b", u"size": 10})
        b = A.from_params(params, name=u"extra")

        assert b.name == u"extra"
        assert b.size == 10

        # Check that extra extras don't get passed.
        params = Params({u"type": u"b", u"size": 10})
        b = A.from_params(params, name=u"extra", unwanted=True)

        assert b.name == u"extra"
        assert b.size == 10

        # Now the same with a custom from_params.
        params = Params({u"type": u"c", u"name": u"extra_c"})
        c = A.from_params(params, size=20)
        assert c.name == u"extra_c"
        assert c.size == 20

        # Check that extra extras don't get passed.
        params = Params({u"type": u"c", u"name": u"extra_c"})
        c = A.from_params(params, size=20, unwanted=True)

        assert c.name == u"extra_c"
        assert c.size == 20

    def test_no_constructor(self):
        params = Params({u"type": u"just_spaces"})

        WordSplitter.from_params(params)
