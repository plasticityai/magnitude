
from __future__ import absolute_import
#overrides

from nltk.sem.logic import TRUTH_TYPE

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.worlds.world import World
from allennlp.semparse import ActionSpaceWalker


class FakeWorldWithAssertions(World):
    # pylint: disable=abstract-method
    #overrides
    def get_valid_starting_types(self):
        return set([TRUTH_TYPE])

    #overrides
    def get_valid_actions(self):
        # This grammar produces true/false statements like
        # (object_exists all_objects)
        # (object_exists (black all_objects))
        # (object_exists (triangle all_objects))
        # (object_exists (touch_wall all_objects))
        # (object_exists (triangle (black all_objects)))
        # ...
        actions = {u'@start@': [u'@start@ -> t'],
                   u't': [u't -> [<o,t>, o]'],
                   u'<o,t>': [u'<o,t> -> object_exists'],
                   u'o': [u'o -> [<o,o>, o]', u'o -> all_objects'],
                   u'<o,o>': [u'<o,o> -> black', u'<o,o> -> triangle', u'<o,o> -> touch_wall']}
        return actions

    #overrides
    def is_terminal(self, symbol     )        :
        return symbol in set([u'object_exists', u'all_objects', u'black', u'triangle', u'touch_wall'])


class ActionSpaceWalkerTest(AllenNlpTestCase):
    def setUp(self):
        super(ActionSpaceWalkerTest, self).setUp()
        self.world = FakeWorldWithAssertions()
        self.walker = ActionSpaceWalker(self.world, max_path_length=10)

    def test_get_logical_forms_with_agenda(self):
        black_logical_forms = self.walker.get_logical_forms_with_agenda([u'<o,o> -> black'])
        # These are all the possible logical forms with black
        assert len(black_logical_forms) == 25
        shortest_logical_form = self.walker.get_logical_forms_with_agenda([u'<o,o> -> black'], 1)[0]
        # This is the shortest complete logical form with black
        assert shortest_logical_form == u'(object_exists (black all_objects))'
        black_triangle_touch_forms = self.walker.get_logical_forms_with_agenda([u'<o,o> -> black',
                                                                                u'<o,o> -> triangle',
                                                                                u'<o,o> -> touch_wall'])
        # Permutations of the three functions. There will not be repetitions of any functions
        # because we limit the length of paths to 10 above.
        assert set(black_triangle_touch_forms) == set([
                u'(object_exists (black (triangle (touch_wall all_objects))))',
                u'(object_exists (black (touch_wall (triangle all_objects))))',
                u'(object_exists (triangle (black (touch_wall all_objects))))',
                u'(object_exists (triangle (touch_wall (black all_objects))))',
                u'(object_exists (touch_wall (black (triangle all_objects))))',
                u'(object_exists (touch_wall (triangle (black all_objects))))'])

    def test_get_all_logical_forms(self):
        # get_all_logical_forms should sort logical forms by length.
        ten_shortest_logical_forms = self.walker.get_all_logical_forms(max_num_logical_forms=10)
        shortest_logical_form = ten_shortest_logical_forms[0]
        assert shortest_logical_form == u'(object_exists all_objects)'
        length_three_logical_forms = ten_shortest_logical_forms[1:4]
        assert set(length_three_logical_forms) == set([u'(object_exists (black all_objects))',
                                                   u'(object_exists (touch_wall all_objects))',
                                                   u'(object_exists (triangle all_objects))'])
