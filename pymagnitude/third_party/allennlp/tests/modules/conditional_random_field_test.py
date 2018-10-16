# pylint: disable=no-self-use,invalid-name


from __future__ import with_statement
from __future__ import absolute_import
import itertools
import math

from pytest import approx, raises
import torch

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
try:
    from itertools import izip
except:
    izip = zip



class TestConditionalRandomField(AllenNlpTestCase):
    def setUp(self):
        super(TestConditionalRandomField, self).setUp()
        self.logits = torch.Tensor([
                [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
                [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ])
        self.tags = torch.LongTensor([
                [2, 3, 4],
                [3, 2, 2]
        ])

        self.transitions = torch.Tensor([
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.8, 0.3, 0.1, 0.7, 0.9],
                [-0.3, 2.1, -5.6, 3.4, 4.0],
                [0.2, 0.4, 0.6, -0.3, -0.4],
                [1.0, 1.0, 1.0, 1.0, 1.0]
        ])

        self.transitions_from_start = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.6])
        self.transitions_to_end = torch.Tensor([-0.1, -0.2, 0.3, -0.4, -0.4])

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = ConditionalRandomField(5)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

    def score(self, logits, tags):
        u"""
        Computes the likelihood score for the given sequence of tags,
        given the provided logits (and the transition weights in the CRF model)
        """
        # Start with transitions from START and to END
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in izip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in izip(logits, tags):
            total += logit[tag]
        return total

    def test_forward_works_without_mask(self):
        log_likelihood = self.crf(self.logits, self.tags).item()

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        # (which is just the score for the logits and actual tags)
        # and the denominator
        # (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i in izip(self.logits, self.tags):
            numerator = self.score(logits_i.detach(), tags_i.detach())
            all_scores = [self.score(logits_i.detach(), tags_j)
                          for tags_j in itertools.product(range(5), repeat=3)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood.item() == approx(log_likelihood)


    def test_forward_works_with_mask(self):
        # Use a non-trivial mask
        mask = torch.LongTensor([
                [1, 1, 1],
                [1, 1, 0]
        ])

        log_likelihood = self.crf(self.logits, self.tags, mask).item()

        # Now compute the log-likelihood manually
        manual_log_likelihood = 0.0

        # For each instance, manually compute the numerator
        #   (which is just the score for the logits and actual tags)
        # and the denominator
        #   (which is the log-sum-exp of the scores for the logits across all possible tags)
        for logits_i, tags_i, mask_i in izip(self.logits, self.tags, mask):
            # Find the sequence length for this input and only look at that much of each sequence.
            sequence_length = torch.sum(mask_i.detach())
            logits_i = logits_i.data[:sequence_length]
            tags_i = tags_i.data[:sequence_length]

            numerator = self.score(logits_i, tags_i)
            all_scores = [self.score(logits_i, tags_j)
                          for tags_j in itertools.product(range(5), repeat=sequence_length)]
            denominator = math.log(sum(math.exp(score) for score in all_scores))
            # And include them in the manual calculation.
            manual_log_likelihood += numerator - denominator

        # The manually computed log likelihood should equal the result of crf.forward.
        assert manual_log_likelihood.item() == approx(log_likelihood)


    def test_viterbi_tags(self):
        mask = torch.LongTensor([
                [1, 1, 1],
                [1, 1, 0]
        ])

        viterbi_path = self.crf.viterbi_tags(self.logits, mask)

        # Separate the tags and scores.
        viterbi_tags = [x for x, y in viterbi_path]
        viterbi_scores = [y for x, y in viterbi_path]

        # Check that the viterbi tags are what I think they should be.
        assert viterbi_tags == [
                [2, 4, 3],
                [4, 2]
        ]

        # We can also iterate over all possible tag sequences and use self.score
        # to check the likelihood of each. The most likely sequence should be the
        # same as what we get from viterbi_tags.
        most_likely_tags = []
        best_scores = []

        for logit, mas in izip(self.logits, mask):
            sequence_length = torch.sum(mas.detach())
            most_likely, most_likelihood = None, -float(u'inf')
            for tags in itertools.product(range(5), repeat=sequence_length):
                score = self.score(logit.data, tags)
                if score > most_likelihood:
                    most_likely, most_likelihood = tags, score
            # Convert tuple to list; otherwise == complains.
            most_likely_tags.append(list(most_likely))
            best_scores.append(most_likelihood)

        assert viterbi_tags == most_likely_tags
        assert viterbi_scores == best_scores

    def test_constrained_viterbi_tags(self):
        constraints = set([(0, 0), (0, 1),
                       (1, 1), (1, 2),
                       (2, 2), (2, 3),
                       (3, 3), (3, 4),
                       (4, 4), (4, 0)])

        # Add the transitions to the end tag
        # and from the start tag.
        for i in range(5):
            constraints.add((5, i))
            constraints.add((i, 6))

        crf = ConditionalRandomField(num_tags=5, constraints=constraints)
        crf.transitions = torch.nn.Parameter(self.transitions)
        crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        mask = torch.LongTensor([
                [1, 1, 1],
                [1, 1, 0]
        ])

        viterbi_path = crf.viterbi_tags(self.logits, mask)

        # Get just the tags from each tuple of (tags, score).
        viterbi_tags = [x for x, y in viterbi_path]

        # Now the tags should respect the constraints
        assert viterbi_tags == [
                [2, 3, 3],
                [2, 3]
        ]

    def test_allowed_transitions(self):
        # pylint: disable=bad-whitespace,bad-continuation
        bio_labels = [u'O', u'B-X', u'I-X', u'B-Y', u'I-Y'] # start tag, end tag
        #              0     1      2      3      4         5          6
        allowed = allowed_transitions(u"BIO", dict(enumerate(bio_labels)))

        # The empty spaces in this matrix indicate disallowed transitions.
        assert set(allowed) == set([                         # Extra column for end tag.
            (0, 0), (0, 1),         (0, 3),              (0, 6),
            (1, 0), (1, 1), (1, 2), (1, 3),              (1, 6),
            (2, 0), (2, 1), (2, 2), (2, 3),              (2, 6),
            (3, 0), (3, 1),         (3, 3), (3, 4),      (3, 6),
            (4, 0), (4, 1),         (4, 3), (4, 4),      (4, 6),
            (5, 0), (5, 1),         (5, 3)])

        bioul_labels = [u'O', u'B-X', u'I-X', u'L-X', u'U-X', u'B-Y', u'I-Y', u'L-Y', u'U-Y'] # start tag, end tag
        #                0     1      2      3      4      5      6      7      8          9        10
        allowed = allowed_transitions(u"BIOUL", dict(enumerate(bioul_labels)))

        # The empty spaces in this matrix indicate disallowed transitions.
        assert set(allowed) == set([                                                   # Extra column for end tag.
            (0, 0), (0, 1),                 (0, 4), (0, 5),                 (0, 8),       (0, 10),
                            (1, 2), (1, 3),
                            (2, 2), (2, 3),
            (3, 0), (3, 1),                 (3, 4), (3, 5),                 (3, 8),       (3, 10),
            (4, 0), (4, 1),                 (4, 4), (4, 5),                 (4, 8),       (4, 10),
                                                            (5, 6), (5, 7),
                                                            (6, 6), (6, 7),
            (7, 0), (7, 1),                 (7, 4), (7, 5),                 (7, 8),       (7, 10),
            (8, 0), (8, 1),                 (8, 4), (8, 5),                 (8, 8),       (8, 10),
            # Extra row for start tag.
            (9, 0), (9, 1),                 (9, 4), (9, 5),                 (9, 8)])

        iob1_labels = [u'O', u'B-X', u'I-X', u'B-Y', u'I-Y'] # start tag, end tag
        #              0     1      2      3      4         5          6
        allowed = allowed_transitions(u"IOB1", dict(enumerate(iob1_labels)))

        # The empty spaces in this matrix indicate disallowed transitions.
        assert set(allowed) == set([                            # Extra column for end tag.
            (0, 0),         (0, 2),         (0, 4),         (0, 6),
            (1, 0), (1, 1), (1, 2),         (1, 4),         (1, 6),
            (2, 0), (2, 1), (2, 2),         (2, 4),         (2, 6),
            (3, 0),         (3, 2), (3, 3), (3, 4),         (3, 6),
            (4, 0),         (4, 2), (4, 3), (4, 4),         (4, 6),
            (5, 0),         (5, 2),         (5, 4),])
        with raises(ConfigurationError):
            allowed_transitions(u"allennlp", {})
