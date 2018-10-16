# pylint: disable=no-self-use,invalid-name

from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.type_declarations import type_declaration as types
from allennlp.semparse.type_declarations.type_declaration import (
        ANY_TYPE,
        BinaryOpType,
        ComplexType,
        NamedBasicType,
        UnaryOpType,
        )
from allennlp.semparse.type_declarations import wikitables_type_declaration as wt_types
from allennlp.semparse.type_declarations.wikitables_type_declaration import (
        CELL_TYPE,
        ROW_TYPE,
        )


class TestTypeDeclaration(AllenNlpTestCase):
    def test_basic_types_conflict_on_names(self):
        type_a = NamedBasicType(u"A")
        type_b = NamedBasicType(u"B")
        assert type_a.resolve(type_b) is None

    def test_unary_ops_resolve_correctly(self):
        unary_type = UnaryOpType()

        # Resolution should fail against a basic type
        assert unary_type.resolve(ROW_TYPE) is None

        # Resolution should fail against a complex type where the argument and return types are not same
        assert unary_type.resolve(ComplexType(CELL_TYPE, ROW_TYPE)) is None

        # Resolution should resolve ANY_TYPE given the other type
        resolution = unary_type.resolve(ComplexType(ANY_TYPE, ROW_TYPE))
        assert resolution == UnaryOpType(ROW_TYPE)
        resolution = unary_type.resolve(ComplexType(CELL_TYPE, ANY_TYPE))
        assert resolution == UnaryOpType(CELL_TYPE)

        reverse_type = ComplexType(ComplexType(CELL_TYPE, ROW_TYPE), ComplexType(CELL_TYPE, ROW_TYPE))
        resolution = unary_type.resolve(reverse_type)
        assert resolution == UnaryOpType(ComplexType(CELL_TYPE, ROW_TYPE))

    def test_binary_ops_resolve_correctly(self):
        binary_type = BinaryOpType()

        # Resolution must fail against a basic type and a complex type that returns a basic type
        assert binary_type.resolve(CELL_TYPE) is None
        assert binary_type.resolve(ComplexType(CELL_TYPE, ROW_TYPE)) is None

        # Resolution must fail against incompatible types
        complex_type = ComplexType(ANY_TYPE, ComplexType(CELL_TYPE, ROW_TYPE))
        assert binary_type.resolve(complex_type) is None

        complex_type = ComplexType(ROW_TYPE, ComplexType(CELL_TYPE, ANY_TYPE))
        assert binary_type.resolve(complex_type) is None

        complex_type = ComplexType(ROW_TYPE, ComplexType(ANY_TYPE, CELL_TYPE))
        assert binary_type.resolve(complex_type) is None

        # Resolution must resolve any types appropriately
        complex_type = ComplexType(ROW_TYPE, ComplexType(ANY_TYPE, ROW_TYPE))
        assert binary_type.resolve(complex_type) == BinaryOpType(ROW_TYPE)

        complex_type = ComplexType(ROW_TYPE, ComplexType(ANY_TYPE, ANY_TYPE))
        assert binary_type.resolve(complex_type) == BinaryOpType(ROW_TYPE)

        complex_type = ComplexType(ANY_TYPE, ComplexType(ROW_TYPE, ANY_TYPE))
        assert binary_type.resolve(complex_type) == BinaryOpType(ROW_TYPE)

    def test_get_valid_actions(self):
        type_r = NamedBasicType(u"R")
        type_d = NamedBasicType(u"D")
        type_e = NamedBasicType(u"E")
        name_mapping = {u'sample_function': u'F'}
        # <e,<r,<d,r>>>
        type_signatures = {u'F': ComplexType(type_e, ComplexType(type_r, ComplexType(type_d, type_r)))}
        basic_types = set([type_r, type_d, type_e])
        valid_actions = types.get_valid_actions(name_mapping, type_signatures, basic_types)
        assert len(valid_actions) == 3
        assert valid_actions[u"<e,<r,<d,r>>>"] == [u"<e,<r,<d,r>>> -> sample_function"]
        assert valid_actions[u"r"] == [u"r -> [<e,<r,<d,r>>>, e, r, d]"]
        assert valid_actions[u"@start@"] == [u"@start@ -> d", u"@start@ -> e", u"@start@ -> r"]

    def test_get_valid_actions_with_placeholder_type(self):
        type_r = NamedBasicType(u"R")
        type_d = NamedBasicType(u"D")
        type_e = NamedBasicType(u"E")
        name_mapping = {u'sample_function': u'F'}
        # <#1,#1>
        type_signatures = {u'F': UnaryOpType()}
        basic_types = set([type_r, type_d, type_e])
        valid_actions = types.get_valid_actions(name_mapping, type_signatures, basic_types)
        assert len(valid_actions) == 5
        assert valid_actions[u"<#1,#1>"] == [u"<#1,#1> -> sample_function"]
        assert valid_actions[u"e"] == [u"e -> [<#1,#1>, e]"]
        assert valid_actions[u"r"] == [u"r -> [<#1,#1>, r]"]
        assert valid_actions[u"d"] == [u"d -> [<#1,#1>, d]"]
        assert valid_actions[u"@start@"] == [u"@start@ -> d", u"@start@ -> e", u"@start@ -> r"]

    def test_get_valid_actions_with_any_type(self):
        type_r = NamedBasicType(u"R")
        type_d = NamedBasicType(u"D")
        type_e = NamedBasicType(u"E")
        name_mapping = {u'sample_function': u'F'}
        # The purpose of this test is to ensure that ANY_TYPE gets substituted by every possible basic type,
        # to simulate an intermediate step while getting actions for a placeholder type.
        # I do not foresee defining a function type with ANY_TYPE. We should just use a ``PlaceholderType``
        # instead.
        # <?,r>
        type_signatures = {u'F': ComplexType(ANY_TYPE, type_r)}
        basic_types = set([type_r, type_d, type_e])
        valid_actions = types.get_valid_actions(name_mapping, type_signatures, basic_types)
        assert len(valid_actions) == 5
        assert valid_actions[u"<d,r>"] == [u"<d,r> -> sample_function"]
        assert valid_actions[u"<e,r>"] == [u"<e,r> -> sample_function"]
        assert valid_actions[u"<r,r>"] == [u"<r,r> -> sample_function"]
        assert valid_actions[u"r"] == [u"r -> [<d,r>, d]", u"r -> [<e,r>, e]", u"r -> [<r,r>, r]"]
        assert valid_actions[u"@start@"] == [u"@start@ -> d", u"@start@ -> e", u"@start@ -> r"]

    def test_get_valid_actions_with_reverse(self):
        valid_actions = types.get_valid_actions(wt_types.COMMON_NAME_MAPPING,
                                                wt_types.COMMON_TYPE_SIGNATURE,
                                                wt_types.BASIC_TYPES)
        assert valid_actions[u'<n,c>'] == [u'<n,c> -> [<<#1,#2>,<#2,#1>>, <c,n>]',
                                          u'<n,c> -> fb:cell.cell.num2',
                                          u'<n,c> -> fb:cell.cell.number']
