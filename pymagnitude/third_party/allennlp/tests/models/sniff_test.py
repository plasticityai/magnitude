# pylint: disable=no-self-use,line-too-long


from __future__ import absolute_import
from __future__ import print_function
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

def demo_model(archive_file     , predictor_name     )             :
    archive = load_archive(archive_file)
    return Predictor.from_archive(archive, predictor_name)

# TODO(joelgrus): this is duplicated in the demo repo
# figure out where it really belongs
DEFAULT_MODELS = {
        u'machine-comprehension': (
                u'https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz',  # pylint: disable=line-too-long
                u'machine-comprehension'
        ),
        u'semantic-role-labeling': (
                u'https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz', # pylint: disable=line-too-long
                u'semantic-role-labeling'
        ),
        u'textual-entailment': (
                u'https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz',  # pylint: disable=line-too-long
                u'textual-entailment'
        ),
        u'coreference-resolution': (
                u'https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz',  # pylint: disable=line-too-long
                u'coreference-resolution'
        ),
        u'named-entity-recognition': (
                u'https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.30.tar.gz',  # pylint: disable=line-too-long
                u'sentence-tagger'
        ),
        u'constituency-parsing': (
                u'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz',  # pylint: disable=line-too-long
                u'constituency-parser'
        ),
        u'dependency-parsing': (
                u'https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-2018.08.01.tar.gz',  # pylint: disable=line-too-long
                u'biaffine-dependency-parser'
        )
}


class SniffTest(AllenNlpTestCase):

    def test_machine_comprehension(self):
        predictor = demo_model(*DEFAULT_MODELS[u'machine-comprehension'])

        passage = u"""The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano. It depicts a dystopian future in which reality as perceived by most humans is actually a simulated reality called "the Matrix", created by sentient machines to subdue the human population, while their bodies' heat and electrical activity are used as an energy source. Computer programmer Neo" learns this truth and is drawn into a rebellion against the machines, which involves other people who have been freed from the "dream world". """  # pylint: disable=line-too-long
        question = u"Who stars in The Matrix?"

        result = predictor.predict_json({u"passage": passage, u"question": question})

        correct = u"Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano"

        assert correct == result[u"best_span_str"]

    def test_semantic_role_labeling(self):
        predictor = demo_model(*DEFAULT_MODELS[u'semantic-role-labeling'])

        sentence = u"If you liked the music we were playing last night, you will absolutely love what we're playing tomorrow!"

        result = predictor.predict_json({u"sentence": sentence})

        assert result[u"words"] == [
                u"If", u"you", u"liked", u"the", u"music", u"we", u"were", u"playing", u"last", u"night", u",",
                u"you", u"will", u"absolutely", u"love", u"what", u"we", u"'re", u"playing", u"tomorrow", u"!"
        ]

        assert result[u"verbs"] == [
                {u"verb": u"liked",
                 u"description": u"If [ARG0: you] [V: liked] [ARG1: the music we were playing last night] , you will absolutely love what we 're playing tomorrow !",
                 u"tags": [u"O", u"B-ARG0", u"B-V", u"B-ARG1", u"I-ARG1", u"I-ARG1", u"I-ARG1", u"I-ARG1", u"I-ARG1", u"I-ARG1", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O"]},
                {u"verb": u"were",
                 u"description": u"If you liked the music we [V: were] playing last night , you will absolutely love what we 're playing tomorrow !",
                 u"tags": [u"O", u"O", u"O", u"O", u"O", u"O", u"B-V", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O"]},
                {u"verb": u"playing",
                 u"description": u"If you liked [ARG1: the music] [ARG0: we] were [V: playing] [ARGM-TMP: last] night , you will absolutely love what we 're playing tomorrow !",
                 u"tags": [u"O", u"O", u"O", u"B-ARG1", u"I-ARG1", u"B-ARG0", u"O", u"B-V", u"B-ARGM-TMP", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O"]},
                {u"verb": u"will",
                 u"description": u"If you liked the music we were playing last night , you [V: will] absolutely love what we 're playing tomorrow !",
                 u"tags": [u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"B-V", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O"]},
                {u"verb": u"love",
                 u"description": u"[ARGM-ADV: If you liked the music we were playing last night] , [ARG0: you] [ARGM-MOD: will] [ARGM-ADV: absolutely] [V: love] [ARG1: what we 're playing tomorrow] !",
                 u"tags": [u"B-ARGM-ADV", u"I-ARGM-ADV", u"I-ARGM-ADV", u"I-ARGM-ADV", u"I-ARGM-ADV", u"I-ARGM-ADV", u"I-ARGM-ADV", u"I-ARGM-ADV", u"I-ARGM-ADV", u"I-ARGM-ADV", u"O", u"B-ARG0", u"B-ARGM-MOD", u"B-ARGM-ADV", u"B-V", u"B-ARG1", u"I-ARG1", u"I-ARG1", u"I-ARG1", u"I-ARG1", u"O"]},
                {u"verb": u"'re",
                 u"description": u"If you liked the music we were playing last night , you will absolutely love what we [V: 're] playing tomorrow !",
                 u"tags": [u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"B-V", u"O", u"O", u"O"]},
                {u"verb": u"playing",
                 u"description": u"If you liked the music we were playing last night , you will absolutely love [ARG1: what] [ARG0: we] 're [V: playing] [ARGM-TMP: tomorrow] !",
                 u"tags": [u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"O", u"B-ARG1", u"B-ARG0", u"O", u"B-V", u"B-ARGM-TMP", u"O"]}
        ]

    def test_textual_entailment(self):
        predictor = demo_model(*DEFAULT_MODELS[u'textual-entailment'])

        result = predictor.predict_json({
                u"premise": u"An interplanetary spacecraft is in orbit around a gas giant's icy moon.",
                u"hypothesis": u"The spacecraft has the ability to travel between planets."
        })

        assert result[u"label_probs"][0] > 0.7  # entailment

        result = predictor.predict_json({
                u"premise": u"Two women are wandering along the shore drinking iced tea.",
                u"hypothesis": u"Two women are sitting on a blanket near some rocks talking about politics."
        })

        assert result[u"label_probs"][1] > 0.8  # contradiction

        result = predictor.predict_json({
                u"premise": u"A large, gray elephant walked beside a herd of zebras.",
                u"hypothesis": u"The elephant was lost."
        })

        assert result[u"label_probs"][2] > 0.6  # neutral

    def test_coreference_resolution(self):
        predictor = demo_model(*DEFAULT_MODELS[u'coreference-resolution'])

        document = u"We 're not going to skimp on quality , but we are very focused to make next year . The only problem is that some of the fabrics are wearing out - since I was a newbie I skimped on some of the fabric and the poor quality ones are developing holes ."

        result = predictor.predict_json({u"document": document})
        print(result)
        assert result[u'clusters'] == [[[0, 0], [10, 10]],
                                      [[33, 33], [37, 37]],
                                      [[26, 27], [42, 43]]]
        assert result[u"document"] == [u'We', u"'re", u'not', u'going', u'to', u'skimp', u'on', u'quality', u',', u'but', u'we', u'are',
                                      u'very', u'focused', u'to', u'make', u'next', u'year', u'.', u'The', u'only', u'problem', u'is',
                                      u'that', u'some', u'of', u'the', u'fabrics', u'are', u'wearing', u'out', u'-', u'since', u'I', u'was',
                                      u'a', u'newbie', u'I', u'skimped', u'on', u'some', u'of', u'the', u'fabric', u'and', u'the', u'poor',
                                      u'quality', u'ones', u'are', u'developing', u'holes', u'.']

    def test_ner(self):
        predictor = demo_model(*DEFAULT_MODELS[u'named-entity-recognition'])

        sentence = u"""Michael Jordan is a professor at Berkeley."""

        result = predictor.predict_json({u"sentence": sentence})

        assert result[u"words"] == [u"Michael", u"Jordan", u"is", u"a", u"professor", u"at", u"Berkeley", u"."]
        assert result[u"tags"] == [u"B-PER", u"L-PER", u"O", u"O", u"O", u"O", u"U-LOC", u"O"]

    def test_constituency_parsing(self):
        predictor = demo_model(*DEFAULT_MODELS[u'constituency-parsing'])

        sentence = u"""Pierre Vinken died aged 81; immortalised aged 61."""

        result = predictor.predict_json({u"sentence": sentence})

        assert result[u"tokens"] == [u"Pierre", u"Vinken", u"died", u"aged", u"81", u";", u"immortalised", u"aged", u"61", u"."]
        assert result[u"trees"] == u"(S (NP (NNP Pierre) (NNP Vinken)) (VP (VP (VBD died) (NP (JJ aged) (CD 81))) (, ;) (VP (VBD immortalised) (S (ADJP (VBN aged) (NP (CD 61)))))) (. .))"

    def test_dependency_parsing(self):
        predictor = demo_model(*DEFAULT_MODELS[u'dependency-parsing'])
        sentence = u"""He ate spaghetti with chopsticks."""
        result = predictor.predict_json({u"sentence": sentence})
        # Note that this tree is incorrect. We are checking here that the decoded
        # tree is _actually a tree_ - in greedy decoding versions of the dependency
        # parser, this sentence has multiple heads. This test shouldn't really live here,
        # but it's very difficult to re-create a concrete example of this behaviour without
        # a trained dependency parser.
        assert result[u'words'] == [u'He', u'ate', u'spaghetti', u'with', u'chopsticks', u'.']
        assert result[u'pos'] == [u'PRP', u'VBD', u'NNS', u'IN', u'NNS', u'.']
        assert result[u'predicted_dependencies'] == [u'nsubj', u'root', u'acomp', u'prep', u'pobj', u'punct']
        assert result[u'predicted_heads'] == [3, 0, 2, 3, 4, 2]
