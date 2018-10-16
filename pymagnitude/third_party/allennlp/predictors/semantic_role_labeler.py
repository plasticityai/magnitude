
from __future__ import absolute_import
#typing

#overrides

from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
try:
    from itertools import izip
except:
    izip = zip



class SemanticRoleLabelerPredictor(Predictor):
    u"""
    Predictor for the :class:`~allennlp.models.bidaf.SemanticRoleLabeler` model.
    """
    def __init__(self, model       , dataset_reader               )        :
        super(SemanticRoleLabelerPredictor, self).__init__(model, dataset_reader)
        self._tokenizer = SpacyWordSplitter(language=u'en_core_web_sm', pos_tags=True)

    def predict(self, sentence     )            :
        u"""
        Predicts the semantic roles of the supplied sentence and returns a dictionary
        with the results.

        .. code-block:: js

            {"words": [...],
             "verbs": [
                {"verb": "...", "description": "...", "tags": [...]},
                ...
                {"verb": "...", "description": "...", "tags": [...]},
            ]}

        Parameters
        ----------
        sentence, ``str``
            The sentence to parse via semantic role labeling.

        Returns
        -------
        A dictionary representation of the semantic roles in the sentence.
        """
        return self.predict_json({u"sentence" : sentence})


    @staticmethod
    def make_srl_string(words           , tags           )       :
        frame = []
        chunk = []

        for (token, tag) in izip(words, tags):
            if tag.startswith(u"I-"):
                chunk.append(token)
            else:
                if chunk:
                    frame.append(u"[" + u" ".join(chunk) + u"]")
                    chunk = []

                if tag.startswith(u"B-"):
                    chunk.append(tag[2:] + u": " + token)
                elif tag == u"O":
                    frame.append(token)

        if chunk:
            frame.append(u"[" + u" ".join(chunk) + u"]")

        return u" ".join(frame)

    #overrides
    def _json_to_instance(self, json_dict          ):
        raise NotImplementedError(u"The SRL model uses a different API for creating instances.")

    def _sentence_to_srl_instances(self, json_dict          )                  :
        u"""
        The SRL model has a slightly different API from other models, as the model is run
        forward for every verb in the sentence. This means that for a single sentence, we need
        to generate a ``List[Instance]``, where the length of this list corresponds to the number
        of verbs in the sentence. Additionally, all of these verbs share the same return dictionary
        after being passed through the model (as really we care about all the frames of the sentence
        together, rather than separately).

        Parameters
        ----------
        json_dict : ``JsonDict``, required.
            JSON that looks like ``{"sentence": "..."}``.

        Returns
        -------
        instances : ``List[Instance]``
            One instance per verb.
        """
        sentence = json_dict[u"sentence"]
        tokens = self._tokenizer.split_words(sentence)
        words = [token.text for token in tokens]
        instances                 = []
        for i, word in enumerate(tokens):
            if word.pos_ == u"VERB":
                verb_labels = [0 for _ in words]
                verb_labels[i] = 1
                instance = self._dataset_reader.text_to_instance(tokens, verb_labels)
                instances.append(instance)
        return instances

    #overrides
    def predict_batch_json(self, inputs                )                  :
        u"""
        Expects JSON that looks like ``[{"sentence": "..."}, {"sentence": "..."}, ...]``
        and returns JSON that looks like

        .. code-block:: js

            [
                {"words": [...],
                 "verbs": [
                    {"verb": "...", "description": "...", "tags": [...]},
                    ...
                    {"verb": "...", "description": "...", "tags": [...]},
                ]},
                {"words": [...],
                 "verbs": [
                    {"verb": "...", "description": "...", "tags": [...]},
                    ...
                    {"verb": "...", "description": "...", "tags": [...]},
                ]}
            ]
        """
        # For SRL, we have more instances than sentences, but the user specified
        # a batch size with respect to the number of sentences passed, so we respect
        # that here by taking the batch size which we use to be the number of sentences
        # we are given.
        batch_size = len(inputs)
        instances_per_sentence = [self._sentence_to_srl_instances(json) for json in inputs]

        flattened_instances = [instance for sentence_instances in instances_per_sentence
                               for instance in sentence_instances]

        if not flattened_instances:
            return sanitize([{u"verbs": [], u"words": self._tokenizer.split_words(x[u"sentence"])}
                             for x in inputs])

        # Make the instances into batches and check the last batch for
        # padded elements as the number of instances might not be perfectly
        # divisible by the batch size.
        batched_instances = group_by_count(flattened_instances, batch_size, None)
        batched_instances[-1] = [instance for instance in batched_instances[-1]
                                 if instance is not None]
        # Run the model on the batches.
        outputs = []
        for batch in batched_instances:
            outputs.extend(self._model.forward_on_instances(batch))

        verbs_per_sentence = [len(sent) for sent in instances_per_sentence]
        return_dicts                 = [{u"verbs": []} for x in inputs]

        output_index = 0
        for sentence_index, verb_count in enumerate(verbs_per_sentence):
            if verb_count == 0:
                # We didn't run any predictions for sentences with no verbs,
                # so we don't have a way to extract the original sentence.
                # Here we just tokenize the input again.
                original_text = self._tokenizer.split_words(inputs[sentence_index][u"sentence"])
                return_dicts[sentence_index][u"words"] = original_text
                continue

            for _ in range(verb_count):
                output = outputs[output_index]
                words = output[u"words"]
                tags = output[u'tags']
                description = self.make_srl_string(words, tags)
                return_dicts[sentence_index][u"words"] = words
                return_dicts[sentence_index][u"verbs"].append({
                        u"verb": output[u"verb"],
                        u"description": description,
                        u"tags": tags,
                })
                output_index += 1

        return sanitize(return_dicts)

    #overrides
    def predict_json(self, inputs          )            :
        u"""
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like

        .. code-block:: js

            {"words": [...],
             "verbs": [
                {"verb": "...", "description": "...", "tags": [...]},
                ...
                {"verb": "...", "description": "...", "tags": [...]},
            ]}
        """
        instances = self._sentence_to_srl_instances(inputs)

        if not instances:
            return sanitize({u"verbs": [], u"words": self._tokenizer.split_words(inputs[u"sentence"])})

        outputs = self._model.forward_on_instances(instances)

        results = {u"verbs": [], u"words": outputs[0][u"words"]}
        for output in outputs:
            tags = output[u'tags']
            description = self.make_srl_string(output[u"words"], tags)
            results[u"verbs"].append({
                    u"verb": output[u"verb"],
                    u"description": description,
                    u"tags": tags,
            })

        return sanitize(results)

SemanticRoleLabelerPredictor = Predictor.register(u"semantic-role-labeling")(SemanticRoleLabelerPredictor)
