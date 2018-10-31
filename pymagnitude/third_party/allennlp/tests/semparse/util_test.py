# pylint: disable=no-self-use,invalid-name

from __future__ import absolute_import
from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse import util


class TestSemparseUtil(AllenNlpTestCase):
    def test_lisp_to_nested_expression(self):
        logical_form = u"((reverse fb:row.row.year) (fb:row.row.league fb:cell.usl_a_league))"
        expression = util.lisp_to_nested_expression(logical_form)
        assert expression == [[[u'reverse', u'fb:row.row.year'], [u'fb:row.row.league', u'fb:cell.usl_a_league']]]
        logical_form = u"(count (and (division 1) (tier (!= null))))"
        expression = util.lisp_to_nested_expression(logical_form)
        assert expression == [[u'count', [u'and', [u'division', u'1'], [u'tier', [u'!=', u'null']]]]]
