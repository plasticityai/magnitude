
from __future__ import absolute_import
#typing


def lisp_to_nested_expression(lisp_string     )        :
    u"""
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack       = []
    current_expression       = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == u'(':
            nested_expression       = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(u')', u''))
        while token[-1] == u')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression
