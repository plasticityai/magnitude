u""" Official evaluation script for v1.1 of the SQuAD dataset. """
# pylint: skip-file
from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
from itertools import imap
from io import open


def normalize_answer(s):
    u"""Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(ur'\b(a|an|the)\b', u' ', text)

    def white_space_fix(text):
        return u' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return u''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article[u'paragraphs']:
            for qa in paragraph[u'qas']:
                total += 1
                if qa[u'id'] not in predictions:
                    message = u'Unanswered question ' + qa[u'id'] +\
                              u' will receive score 0.'
                    print >>sys.stderr, message
                    continue
                ground_truths = list(imap(lambda x: x[u'text'], qa[u'answers']))
                prediction = predictions[qa[u'id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {u'exact_match': exact_match, u'f1': f1}


if __name__ == u'__main__':
    expected_version = u'1.1'
    parser = argparse.ArgumentParser(
        description=u'Evaluation for SQuAD ' + expected_version)
    parser.add_argument(u'dataset_file', help=u'Dataset file')
    parser.add_argument(u'prediction_file', help=u'Prediction File')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json[u'version'] != expected_version):
            print >>sys.stderr, u'Evaluation expects v-' + expected_version +
                  u', but got dataset with v-' + dataset_json[u'version']
        dataset = dataset_json[u'data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    print json.dumps(evaluate(dataset, predictions))
