u"""
We define a simple deterministic decoder here, that takes steps to add integers to list. At
each step, the decoder takes the last integer in the list, and adds either 1 or 2 to produce the
next element that will be added to the list. We initialize the list with the value 0 (or whatever
you pick), and we say that a sequence is finished when the last element is 4. We define the score
of a state as the negative of the number of elements (excluding the initial value) in the action
history.
"""

from __future__ import absolute_import
from collections import defaultdict
#typing

#overrides
import torch

from allennlp.nn.decoding import DecoderState, DecoderStep
try:
    from itertools import izip
except:
    izip = zip


class SimpleDecoderState(DecoderState[u'SimpleDecoderState']):
    def __init__(self,
                 batch_indices           ,
                 action_history                 ,
                 score                    ,
                 start_values            = None)        :
        super(SimpleDecoderState, self).__init__(batch_indices, action_history, score)
        self.start_values = start_values or [0] * len(batch_indices)

    def is_finished(self)        :
        return self.action_history[0][-1] == 4

    @classmethod
    def combine_states(cls, states)                        :
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in
                            state.action_history]
        scores = [score for state in states for score in state.score]
        start_values = [start_value for state in states for start_value in state.start_values]
        return SimpleDecoderState(batch_indices, action_histories, scores, start_values)

    def __repr__(self):
        return "{self.action_history}"


class SimpleDecoderStep(DecoderStep[SimpleDecoderState]):
    def __init__(self,
                 valid_actions           = None,
                 include_value_in_score       = False)        :
        # The default allowed actions are adding 1 or 2 to the last element.
        self._valid_actions = valid_actions or set([1, 2])
        # If True, we will add a small multiple of the action take to the score, to encourage
        # getting higher numbers first (and to differentiate action sequences).
        self._include_value_in_score = include_value_in_score

    #overrides
    def take_step(self,
                  state                    ,
                  max_actions      = None,
                  allowed_actions            = None)                            :
        indexed_next_states                                      = defaultdict(list)
        if not allowed_actions:
            allowed_actions = [None] * len(state.batch_indices)
        for batch_index, action_history, score, start_value, actions in izip(state.batch_indices,
                                                                            state.action_history,
                                                                            state.score,
                                                                            state.start_values,
                                                                            allowed_actions):

            prev_action = action_history[-1] if action_history else start_value
            for action in self._valid_actions:
                next_item = int(prev_action + action)
                if actions and next_item not in actions:
                    continue
                new_history = action_history + [next_item]
                # For every action taken, we reduce the score by 1.
                new_score = score - 1
                if self._include_value_in_score:
                    new_score += 0.01 * next_item
                new_state = SimpleDecoderState([batch_index],
                                               [new_history],
                                               [new_score])
                indexed_next_states[batch_index].append(new_state)
        next_states                           = []
        for batch_next_states in list(indexed_next_states.values()):
            sorted_next_states = [(-state.score[0].data[0], state) for state in batch_next_states]
            sorted_next_states.sort(key=lambda x: x[0])
            if max_actions is not None:
                sorted_next_states = sorted_next_states[:max_actions]
            next_states.extend(state[1] for state in sorted_next_states)
        return next_states
