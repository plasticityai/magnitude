
from __future__ import absolute_import
#typing
#overrides

from nltk.sem.logic import TRUTH_TYPE, BasicType, EntityType, Type

from allennlp.semparse.type_declarations.type_declaration import ComplexType, HigherOrderType, NamedBasicType


class NegateFilterType(HigherOrderType):
    u"""
    Because our negate filters are higher-order functions, we need to make an explicit class here,
    to make sure that we've overridden the right methods correctly.
    """
    def __init__(self, first, second):
        super(NegateFilterType, self).__init__(num_arguments=1, first=first, second=second)

    #overrides
    def substitute_any_type(self, basic_types                )              :
        # There's no ANY_TYPE in here, so we don't need to do any substitution.
        return [self]


# All constants default to ``EntityType`` in NLTK. For domains where constants of different types
# appear in the logical forms, we have a way of specifying ``constant_type_prefixes`` and passing
# them to the constructor of ``World``. However, in the NLVR language we defined, we see constants
# of just one type, number. So we let them default to ``EntityType``.
NUM_TYPE = EntityType()
BOX_TYPE = NamedBasicType(u"BOX")
OBJECT_TYPE = NamedBasicType(u"OBJECT")
COLOR_TYPE = NamedBasicType(u"COLOR")
SHAPE_TYPE = NamedBasicType(u"SHAPE")
OBJECT_FILTER_TYPE = ComplexType(OBJECT_TYPE, OBJECT_TYPE)
NEGATE_FILTER_TYPE = NegateFilterType(ComplexType(OBJECT_TYPE, OBJECT_TYPE),
                                      ComplexType(OBJECT_TYPE, OBJECT_TYPE))
BOX_MEMBERSHIP_TYPE = ComplexType(BOX_TYPE, OBJECT_TYPE)

BOX_COLOR_FILTER_TYPE = ComplexType(BOX_TYPE, ComplexType(COLOR_TYPE, BOX_TYPE))
BOX_SHAPE_FILTER_TYPE = ComplexType(BOX_TYPE, ComplexType(SHAPE_TYPE, BOX_TYPE))
BOX_COUNT_FILTER_TYPE = ComplexType(BOX_TYPE, ComplexType(NUM_TYPE, BOX_TYPE))
# This box filter returns boxes where a specified attribute is same or different
BOX_ATTRIBUTE_SAME_FILTER_TYPE = ComplexType(BOX_TYPE, BOX_TYPE)


ASSERT_COLOR_TYPE = ComplexType(OBJECT_TYPE, ComplexType(COLOR_TYPE, TRUTH_TYPE))
ASSERT_SHAPE_TYPE = ComplexType(OBJECT_TYPE, ComplexType(SHAPE_TYPE, TRUTH_TYPE))
ASSERT_BOX_COUNT_TYPE = ComplexType(BOX_TYPE, ComplexType(NUM_TYPE, TRUTH_TYPE))
ASSERT_OBJECT_COUNT_TYPE = ComplexType(OBJECT_TYPE, ComplexType(NUM_TYPE, TRUTH_TYPE))

BOX_EXISTS_TYPE = ComplexType(BOX_TYPE, TRUTH_TYPE)
OBJECT_EXISTS_TYPE = ComplexType(OBJECT_TYPE, TRUTH_TYPE)


COMMON_NAME_MAPPING = {}
COMMON_TYPE_SIGNATURE = {}

BASIC_TYPES = set([NUM_TYPE, BOX_TYPE, OBJECT_TYPE, COLOR_TYPE, SHAPE_TYPE])


def add_common_name_with_type(name, mapping, type_signature):
    COMMON_NAME_MAPPING[name] = mapping
    COMMON_TYPE_SIGNATURE[mapping] = type_signature


# Entities
add_common_name_with_type(u"all_objects", u"O", OBJECT_TYPE)
add_common_name_with_type(u"all_boxes", u"B", BOX_TYPE)
add_common_name_with_type(u"color_black", u"C0", COLOR_TYPE)
add_common_name_with_type(u"color_blue", u"C1", COLOR_TYPE)
add_common_name_with_type(u"color_yellow", u"C2", COLOR_TYPE)
add_common_name_with_type(u"shape_triangle", u"S0", SHAPE_TYPE)
add_common_name_with_type(u"shape_square", u"S1", SHAPE_TYPE)
add_common_name_with_type(u"shape_circle", u"S2", SHAPE_TYPE)


# Attribute function
add_common_name_with_type(u"object_in_box", u"I", BOX_MEMBERSHIP_TYPE)


# Assert functions
add_common_name_with_type(u"object_color_all_equals", u"A0", ASSERT_COLOR_TYPE)
add_common_name_with_type(u"object_color_any_equals", u"A28", ASSERT_COLOR_TYPE)
add_common_name_with_type(u"object_color_none_equals", u"A1", ASSERT_COLOR_TYPE)
add_common_name_with_type(u"object_shape_all_equals", u"A2", ASSERT_SHAPE_TYPE)
add_common_name_with_type(u"object_shape_any_equals", u"A29", ASSERT_SHAPE_TYPE)
add_common_name_with_type(u"object_shape_none_equals", u"A3", ASSERT_SHAPE_TYPE)
add_common_name_with_type(u"box_count_equals", u"A4", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type(u"box_count_not_equals", u"A5", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type(u"box_count_greater", u"A6", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type(u"box_count_greater_equals", u"A7", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type(u"box_count_lesser", u"A8", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type(u"box_count_lesser_equals", u"A9", ASSERT_BOX_COUNT_TYPE)
add_common_name_with_type(u"object_count_equals", u"A10", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_count_not_equals", u"A11", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_count_greater", u"A12", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_count_greater_equals", u"A13", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_count_lesser", u"A14", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_count_lesser_equals", u"A15", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_color_count_equals", u"A16", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_color_count_not_equals", u"A17", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_color_count_greater", u"A18", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_color_count_greater_equals", u"A19", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_color_count_lesser", u"A20", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_color_count_lesser_equals", u"A21", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_shape_count_equals", u"A22", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_shape_count_not_equals", u"A23", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_shape_count_greater", u"A24", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_shape_count_greater_equals", u"A25", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_shape_count_lesser", u"A26", ASSERT_OBJECT_COUNT_TYPE)
add_common_name_with_type(u"object_shape_count_lesser_equals", u"A27", ASSERT_OBJECT_COUNT_TYPE)

add_common_name_with_type(u"box_exists", u"E0", BOX_EXISTS_TYPE)
add_common_name_with_type(u"object_exists", u"E1", OBJECT_EXISTS_TYPE)


# Box filter functions
add_common_name_with_type(u"member_count_equals", u"F0", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_count_not_equals", u"F1", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_shape_all_equals", u"F2", BOX_SHAPE_FILTER_TYPE)
add_common_name_with_type(u"member_shape_any_equals", u"F26", BOX_SHAPE_FILTER_TYPE)
add_common_name_with_type(u"member_shape_none_equals", u"F3", BOX_SHAPE_FILTER_TYPE)
add_common_name_with_type(u"member_color_all_equals", u"F4", BOX_COLOR_FILTER_TYPE)
add_common_name_with_type(u"member_color_any_equals", u"F27", BOX_COLOR_FILTER_TYPE)
add_common_name_with_type(u"member_color_none_equals", u"F5", BOX_COLOR_FILTER_TYPE)
add_common_name_with_type(u"member_count_greater", u"F6", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_count_greater_equals", u"F7", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_count_lesser", u"F8", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_count_lesser_equals", u"F9", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_color_count_equals", u"F10", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_color_count_not_equals", u"F11", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_color_count_greater", u"F12", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_color_count_greater_equals", u"F13", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_color_count_lesser", u"F14", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_color_count_lesser_equals", u"F15", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_shape_count_equals", u"F16", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_shape_count_not_equals", u"F17", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_shape_count_greater", u"F18", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_shape_count_greater_equals", u"F19", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_shape_count_lesser", u"F20", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_shape_count_lesser_equals", u"F21", BOX_COUNT_FILTER_TYPE)
add_common_name_with_type(u"member_shape_same", u"F22", BOX_ATTRIBUTE_SAME_FILTER_TYPE)
add_common_name_with_type(u"member_color_same", u"F23", BOX_ATTRIBUTE_SAME_FILTER_TYPE)
add_common_name_with_type(u"member_shape_different", u"F24", BOX_ATTRIBUTE_SAME_FILTER_TYPE)
add_common_name_with_type(u"member_color_different", u"F25", BOX_ATTRIBUTE_SAME_FILTER_TYPE)


# Object filter functions
add_common_name_with_type(u"black", u"C3", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"blue", u"C4", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"yellow", u"C5", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"same_color", u"C6", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"triangle", u"S3", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"square", u"S4", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"circle", u"S5", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"same_shape", u"S6", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"touch_wall", u"T0", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"touch_corner", u"T1", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"touch_top", u"T2", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"touch_bottom", u"T3", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"touch_left", u"T4", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"touch_right", u"T5", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"touch_object", u"T6", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"above", u"L0", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"below", u"L1", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"top", u"L2", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"bottom", u"L3", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"small", u"Z0", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"medium", u"Z1", OBJECT_FILTER_TYPE)
add_common_name_with_type(u"big", u"Z2", OBJECT_FILTER_TYPE)

add_common_name_with_type(u"negate_filter", u"N", NEGATE_FILTER_TYPE)

# Adding numbers because they commonly occur in utterances. They're usually between 1 and 9. Since
# there are not too many of these productions, we're adding them to the global mapping instead of a
# local mapping in each world.
for num in range(1, 10):
    num_string = unicode(num)
    add_common_name_with_type(num_string, num_string, NUM_TYPE)
