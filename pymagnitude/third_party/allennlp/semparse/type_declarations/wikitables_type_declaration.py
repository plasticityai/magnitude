u"""
Defines all the types in the WikitablesQuestions domain.
"""

from __future__ import absolute_import
#typing
#overrides

from nltk.sem.logic import Type, BasicType, ANY_TYPE, ComplexType as NltkComplexType

from allennlp.semparse.type_declarations.type_declaration import ComplexType, HigherOrderType
from allennlp.semparse.type_declarations.type_declaration import PlaceholderType, NamedBasicType
from allennlp.semparse.type_declarations.type_declaration import UnaryOpType, BinaryOpType


class ReverseType(PlaceholderType, HigherOrderType):
    u"""
    ReverseType is a kind of ``PlaceholderType`` where type resolution involves matching the return
    type with the reverse of the argument type. So all we care about are the types of the surrounding
    expressions, and return a resolution that matches whatever parts are present in the type signatures
    of the arguments and the return expressions.

    Following are the resolutions for some example type signatures being matched against::

        <?, <e,r>>      :   <<r,e>, <e,r>>
        <<r,?>, <e,?>>  :   <<r,e>, <e,r>>
        <<r,?>, ?>      :   <<r,?>, <?,r>>>
        <<r,?>, <?,e>>  :   None
    """
    def __init__(self, first      , second      )        :
        super(ReverseType, self).__init__(num_arguments=1, first=first, second=second)
        self._signature = u'<<#1,#2>,<#2,#1>>'

    #overrides
    def resolve(self, other      )                  :
        # Idea: Since its signature is <<#1,#2>,<#2,#1>> no information about types in self is
        # relevant.  All that matters is that other.first resolves against the reverse of
        # other.second and vice versa.
        if not isinstance(other, NltkComplexType):
            return None
        # other.first and other.second are the argument and return types respectively.
        reversed_second = ComplexType(other.second.second, other.second.first)
        other_first = other.first.resolve(reversed_second)
        if not other_first:
            return None
        reversed_first = ComplexType(other_first.second, other_first.first)
        other_second = other.second.resolve(reversed_first)
        if not other_second:
            return None
        return ReverseType(other_first, other_second)

    #overrides
    def get_application_type(self, argument_type      )        :
        return ComplexType(argument_type.second, argument_type.first)

    #overrides
    def substitute_any_type(self, basic_types                )              :
        basic_first_types = basic_types if self.first.first == ANY_TYPE else set([self.first.first])
        basic_second_types = basic_types if self.first.second == ANY_TYPE else set([self.first.second])
        substitutions = []
        for first_type in basic_first_types:
            for second_type in basic_second_types:
                substituted_first = ComplexType(first_type, second_type)
                substituted_second = ComplexType(second_type, first_type)
                substitutions.append(ReverseType(substituted_first, substituted_second))
        return substitutions


class ArgExtremeType(PlaceholderType):
    u"""
    This is the type for argmax and argmin in Sempre. The type signature is <d,<d,<#1,<<d,#1>,#1>>>>.
    Example: (argmax (number 1) (number 1) (fb:row.row.league fb:cell.usl_a_league) fb:row.row.index)
    meaning, of the subset of rows where league == usl_a_league, find the row with the maximum index.
    """
    def __init__(self, basic_type            = ANY_TYPE, lambda_arg_type            = ANY_TYPE)        :
        super(ArgExtremeType, self).__init__(NUMBER_TYPE,
                         ComplexType(NUMBER_TYPE,
                                     ComplexType(basic_type,
                                                 ComplexType(ComplexType(lambda_arg_type, basic_type),
                                                             basic_type))))
        self._signature = u'<n,<n,<#1,<<#2,#1>,#1>>>>'

    #overrides
    def resolve(self, other      )                  :
        u"""See ``PlaceholderType.resolve``"""
        if not isinstance(other, NltkComplexType):
            return None
        expected_second = ComplexType(NUMBER_TYPE,
                                      ComplexType(ANY_TYPE, ComplexType(ComplexType(ANY_TYPE, ANY_TYPE),
                                                                        ANY_TYPE)))
        resolved_second = other.second.resolve(expected_second)
        if resolved_second is None:
            return None

        # The lambda function that we use inside the argmax  must take either a number or a date as
        # an argument.
        lambda_arg_type = other.second.second.second.first.first
        if lambda_arg_type.resolve(NUMBER_TYPE) is None and lambda_arg_type.resolve(DATE_TYPE) is None:
            return None

        try:
            # This is the first #1 in the type signature above.
            selector_function_type = resolved_second.second.first
            # This is the second #1 in the type signature above.
            quant_function_argument_type = resolved_second.second.second.first.second
            # This is the third #1 in the type signature above.
            return_type = resolved_second.second.second.second

            # All three placeholder (ph) types above should resolve against each other.
            resolved_first_ph = selector_function_type.resolve(quant_function_argument_type)
            resolved_first_ph.resolve(return_type)

            resolved_second_ph = quant_function_argument_type.resolve(resolved_first_ph)
            resolved_second_ph.resolve(return_type)

            resolved_third_ph = return_type.resolve(resolved_first_ph)
            resolved_third_ph = return_type.resolve(resolved_second_ph)

            if not resolved_first_ph or not resolved_second_ph or not resolved_third_ph:
                return None

            return ArgExtremeType(resolved_first_ph, lambda_arg_type)
        except AttributeError:
            return None

    #overrides
    def get_application_type(self, argument_type      )        :
        # Returning <d,<#1,<<d,#1>,#1>>>.
        # This is called after the placeholders are resolved.
        return self.second

    #overrides
    def substitute_any_type(self, basic_types                )              :
        if self.second.second.first != ANY_TYPE:
            return [self]
        return [ArgExtremeType(basic_type, inner_function_type)
                for basic_type in basic_types
                for inner_function_type in set([NUMBER_TYPE, DATE_TYPE])]


class CountType(PlaceholderType):
    u"""
    Type of a function that counts arbitrary things. Signature is <#1,n>.
    """
    def __init__(self, count_type      )        :
        super(CountType, self).__init__(count_type, NUMBER_TYPE)
        self._signature = u'<#1,n>'

    #overrides
    def resolve(self, other      )        :
        u"""See ``PlaceholderType.resolve``"""
        if not isinstance(other, NltkComplexType):
            return None
        resolved_second = NUMBER_TYPE.resolve(other.second)
        if not resolved_second:
            return None
        return CountType(other.first)

    #overrides
    def get_application_type(self, argument_type      )        :
        return NUMBER_TYPE

    #overrides
    def substitute_any_type(self, basic_types                )              :
        if self.first != ANY_TYPE:
            return [self]
        return [CountType(basic_type) for basic_type in basic_types]


CELL_TYPE = NamedBasicType(u"CELL")
PART_TYPE = NamedBasicType(u"PART")
ROW_TYPE = NamedBasicType(u"ROW")
DATE_TYPE = NamedBasicType(u"DATE")
NUMBER_TYPE = NamedBasicType(u"NUMBER")

BASIC_TYPES = set([CELL_TYPE, PART_TYPE, ROW_TYPE, DATE_TYPE, NUMBER_TYPE])
# Functions like fb:row.row.year.
COLUMN_TYPE = ComplexType(CELL_TYPE, ROW_TYPE)
# fb:cell.cell.part
PART_TO_CELL_TYPE = ComplexType(PART_TYPE, CELL_TYPE)
# fb:cell.cell.date
DATE_TO_CELL_TYPE = ComplexType(DATE_TYPE, CELL_TYPE)
# fb:cell.cell.number
NUM_TO_CELL_TYPE = ComplexType(NUMBER_TYPE, CELL_TYPE)
# number
NUMBER_FUNCTION_TYPE = ComplexType(NUMBER_TYPE, NUMBER_TYPE)
# date (Signature: <e,<e,<e,d>>>; Example: (date 1982 -1 -1))
DATE_FUNCTION_TYPE = ComplexType(NUMBER_TYPE,
                                 ComplexType(NUMBER_TYPE, ComplexType(NUMBER_TYPE, DATE_TYPE)))
# Unary numerical operations: max, min, >, <, sum etc.
UNARY_DATE_NUM_OP_TYPE = UnaryOpType(allowed_substitutions=set([DATE_TYPE, NUMBER_TYPE]),
                                     signature=u'<nd,nd>')
UNARY_NUM_OP_TYPE = ComplexType(NUMBER_TYPE, NUMBER_TYPE)

# Binary numerical operation: -
BINARY_NUM_OP_TYPE = ComplexType(NUMBER_TYPE, ComplexType(NUMBER_TYPE, NUMBER_TYPE))

# next
ROW_TO_ROW_TYPE = ComplexType(ROW_TYPE, ROW_TYPE)
# reverse
REVERSE_TYPE = ReverseType(ComplexType(ANY_TYPE, ANY_TYPE), ComplexType(ANY_TYPE, ANY_TYPE))
# !=, fb:type.object.type
# fb:type.object.type takes a type and returns all objects of that type.
IDENTITY_TYPE = UnaryOpType()
# index
ROW_INDEX_TYPE = ComplexType(NUMBER_TYPE, ROW_TYPE)
# count
COUNT_TYPE = CountType(ANY_TYPE)
# and, or
CONJUNCTION_TYPE = BinaryOpType()
# argmax, argmin
ARG_EXTREME_TYPE = ArgExtremeType()


COMMON_NAME_MAPPING = {u"lambda": u"\\", u"var": u"V", u"x": u"X"}


COMMON_TYPE_SIGNATURE = {u"V": IDENTITY_TYPE, u"X": ANY_TYPE}


def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature

add_common_name_with_type(u"reverse", u"R", REVERSE_TYPE)
add_common_name_with_type(u"argmax", u"A0", ARG_EXTREME_TYPE)
add_common_name_with_type(u"argmin", u"A1", ARG_EXTREME_TYPE)
add_common_name_with_type(u"max", u"M0", UNARY_DATE_NUM_OP_TYPE)
add_common_name_with_type(u"min", u"M1", UNARY_DATE_NUM_OP_TYPE)
add_common_name_with_type(u"and", u"A", CONJUNCTION_TYPE)
add_common_name_with_type(u"or", u"O", CONJUNCTION_TYPE)
add_common_name_with_type(u"fb:row.row.next", u"N", ROW_TO_ROW_TYPE)
add_common_name_with_type(u"number", u"I", NUMBER_FUNCTION_TYPE)
add_common_name_with_type(u"date", u"D0", DATE_FUNCTION_TYPE)
add_common_name_with_type(u"fb:cell.cell.part", u"P", PART_TO_CELL_TYPE)
add_common_name_with_type(u"fb:cell.cell.date", u"D1", DATE_TO_CELL_TYPE)
add_common_name_with_type(u"fb:cell.cell.number", u"I1", NUM_TO_CELL_TYPE)
add_common_name_with_type(u"fb:cell.cell.num2", u"I2", NUM_TO_CELL_TYPE)
add_common_name_with_type(u"fb:row.row.index", u"W", ROW_INDEX_TYPE)
add_common_name_with_type(u"fb:type.row", u"T0", ROW_TYPE)
add_common_name_with_type(u"fb:type.object.type", u"T", ROW_TO_ROW_TYPE)
add_common_name_with_type(u"count", u"C", COUNT_TYPE)
add_common_name_with_type(u"!=", u"Q", IDENTITY_TYPE)
add_common_name_with_type(u">", u"G0", UNARY_DATE_NUM_OP_TYPE)
add_common_name_with_type(u">=", u"G1", UNARY_DATE_NUM_OP_TYPE)
add_common_name_with_type(u"<", u"L0", UNARY_DATE_NUM_OP_TYPE)
add_common_name_with_type(u"<=", u"L1", UNARY_DATE_NUM_OP_TYPE)
add_common_name_with_type(u"sum", u"S0", UNARY_NUM_OP_TYPE)
add_common_name_with_type(u"avg", u"S1", UNARY_NUM_OP_TYPE)
add_common_name_with_type(u"-", u"F", BINARY_NUM_OP_TYPE)  # subtraction
