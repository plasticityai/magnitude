
from __future__ import absolute_import
#typing

#overrides

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token


class WordFilter(Registrable):
    u"""
    A ``WordFilter`` removes words from a token list.  Typically, this is for stopword removal,
    though you could feasibly use it for more domain-specific removal if you want.

    Word removal happens `before` stemming, so keep that in mind if you're designing a list of
    words to be removed.
    """
    default_implementation = u'pass_through'

    def filter_words(self, words             )               :
        u"""
        Returns a filtered list of words.
        """
        raise NotImplementedError


class PassThroughWordFilter(WordFilter):
    u"""
    Does not filter words; it's a no-op.  This is the default word filter.
    """
    #overrides
    def filter_words(self, words             )               :
        return words


PassThroughWordFilter = WordFilter.register(u'pass_through')(PassThroughWordFilter)

class StopwordFilter(WordFilter):
    u"""
    Uses a list of stopwords to filter.
    """
    def __init__(self):
        # TODO(matt): Allow this to be specified somehow, either with a file, or with parameters,
        # or something.
        self.stopwords = set([u'I', u'a', u'aboard', u'about', u'above', u'accordance', u'according',
                              u'across', u'after', u'against', u'along', u'alongside', u'also', u'am',
                              u'amid', u'amidst', u'an', u'and', u'apart', u'are', u'around', u'as',
                              u'aside', u'astride', u'at', u'atop', u'back', u'be', u'because', u'before',
                              u'behind', u'below', u'beneath', u'beside', u'besides', u'between',
                              u'beyond', u'but', u'by', u'concerning', u'do', u'down', u'due', u'during',
                              u'either', u'except', u'exclusive', u'false', u'for', u'from', u'happen',
                              u'he', u'her', u'hers', u'herself', u'him', u'himself', u'his', u'how',
                              u'how many', u'how much', u'i', u'if', u'in', u'including', u'inside',
                              u'instead', u'into', u'irrespective', u'is', u'it', u'its', u'itself',
                              u'less', u'me', u'mine', u'minus', u'my', u'myself', u'neither', u'next',
                              u'not', u'occur', u'of', u'off', u'on', u'onto', u'opposite', u'or', u'our',
                              u'ours', u'ourselves', u'out', u'out of', u'outside', u'over', u'owing',
                              u'per', u'prepatory', u'previous', u'prior', u'pursuant', u'regarding',
                              u's', u'sans', u'she', u'subsequent', u'such', u'than', u'thanks', u'that',
                              u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'these',
                              u'they', u'this', u'those', u'through', u'throughout', u'thru', u'till',
                              u'to', u'together', u'top', u'toward', u'towards', u'true', u'under',
                              u'underneath', u'unlike', u'until', u'up', u'upon', u'us', u'using',
                              u'versus', u'via', u'was', u'we', u'were', u'what', u'when', u'where',
                              u'which', u'who', u'why', u'will', u'with', u'within', u'without', u'you',
                              u'your', u'yours', u'yourself', u'yourselves', u",", u'.', u':', u'!', u';',
                              u"'", u'"', u'&', u'$', u'#', u'@', u'(', u')', u'?'])

    #overrides
    def filter_words(self, words             )               :
        return [word for word in words if word.text.lower() not in self.stopwords]

StopwordFilter = WordFilter.register(u'stopwords')(StopwordFilter)
