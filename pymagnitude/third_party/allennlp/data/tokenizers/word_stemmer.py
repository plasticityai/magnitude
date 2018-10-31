
from __future__ import absolute_import
from nltk.stem import PorterStemmer as NltkPorterStemmer
#overrides

from allennlp.common import Registrable
from allennlp.data.tokenizers.token import Token


class WordStemmer(Registrable):
    u"""
    A ``WordStemmer`` lemmatizes words.  This means that we map words to their root form, so that,
    e.g., "have", "has", and "had" all have the same internal representation.

    You should think carefully about whether and how much stemming you want in your model.  Kind of
    the whole point of using word embeddings is so that you don't have to do this, but in a highly
    inflected language, or in a low-data setting, you might need it anyway.  The default
    ``WordStemmer`` does nothing, just returning the work token as-is.
    """
    default_implementation = u'pass_through'

    def stem_word(self, word       )         :
        u"""
        Returns a new ``Token`` with ``word.text`` replaced by a stemmed word.
        """
        raise NotImplementedError


class PassThroughWordStemmer(WordStemmer):
    u"""
    Does not stem words; it's a no-op.  This is the default word stemmer.
    """
    #overrides
    def stem_word(self, word       )         :
        return word


PassThroughWordStemmer = WordStemmer.register(u'pass_through')(PassThroughWordStemmer)

class PorterStemmer(WordStemmer):
    u"""
    Uses NLTK's PorterStemmer to stem words.
    """
    def __init__(self):
        self.stemmer = NltkPorterStemmer()

    #overrides
    def stem_word(self, word       )         :
        new_text = self.stemmer.stem(word.text)
        return Token(text=new_text,
                     idx=word.idx,
                     lemma=word.lemma_,
                     pos=word.pos_,
                     tag=word.tag_,
                     dep=word.dep_,
                     ent_type=word.ent_type_,
                     text_id=getattr(word, u'text_id', None))

PorterStemmer = WordStemmer.register(u'porter')(PorterStemmer)
