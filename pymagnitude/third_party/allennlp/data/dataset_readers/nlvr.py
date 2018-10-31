

from __future__ import with_statement
from __future__ import absolute_import
#typing
import json
import logging

#overrides

from allennlp.common.util import JsonDict
from allennlp.data.instance import Instance
from allennlp.data.fields import Field, TextField, ListField, IndexField, LabelField
from allennlp.data.fields import ProductionRuleField, MetadataField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.semparse.worlds import NlvrWorld
from io import open


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NlvrDatasetReader(DatasetReader):
    u"""
    ``DatasetReader`` for the NLVR domain. In addition to the usual methods for reading files and
    instances from text, this class contains a method for creating an agenda of actions that each
    sentence triggers, if needed. Note that we deal with the version of the dataset with structured
    representations of the synthetic images instead of the actual images themselves.

    We support multiple data formats here:
    1) The original json version of the NLVR dataset (http://lic.nlp.cornell.edu/nlvr/) where the
    format of each line in the jsonl file is
    ```
    "sentence": <sentence>,
    "label": <true/false>,
    "identifier": <id>,
    "evals": <dict containing all annotations>,
    "structured_rep": <list of three box representations, where each box is a list of object
    representation dicts, containing fields "x_loc", "y_loc", "color", "type", "size">
    ```

    2) A grouped version (constructed using ``scripts/nlvr/group_nlvr_worlds.py``) where we group
    all the worlds that a sentence appears in. We use the fields ``sentence``, ``label`` and
    ``structured_rep``.  And the format of the grouped files is
    ```
    "sentence": <sentence>,
    "labels": <list of labels corresponding to worlds the sentence appears in>
    "identifier": <id that is only the prefix from the original data>
    "worlds": <list of structured representations>
    ```

    3) A processed version that contains action sequences that lead to the correct denotations (or
    not), using some search. This format is very similar to the grouped format, and has the
    following extra field

    ```
    "correct_sequences": <list of lists of action sequences corresponding to logical forms that
    evaluate to the correct denotations>
    ```

    Parameters
    ----------
    lazy : ``bool`` (optional, default=False)
        Passed to ``DatasetReader``.  If this is ``True``, training will start sooner, but will
        take longer per batch.
    tokenizer : ``Tokenizer`` (optional)
        The tokenizer used for sentences in NLVR. Default is ``WordTokenizer``
    sentence_token_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Token indexers for tokens in input sentences.
        Default is ``{"tokens": SingleIdTokenIndexer()}``
    nonterminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for non-terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    terminal_indexers : ``Dict[str, TokenIndexer]`` (optional)
        Indexers for terminals in production rules. The default is to index terminals and
        non-terminals in the same way, but you may want to change it.
        Default is ``{"tokens": SingleIdTokenIndexer("rule_labels")}``
    output_agendas : ``bool`` (optional)
        If preparing data for a trainer that uses agendas, set this flag and the datset reader will
        output agendas.
    """
    def __init__(self,
                 lazy       = False,
                 tokenizer            = None,
                 sentence_token_indexers                          = None,
                 nonterminal_indexers                          = None,
                 terminal_indexers                          = None,
                 output_agendas       = True)        :
        super(NlvrDatasetReader, self).__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._sentence_token_indexers = sentence_token_indexers or {u"tokens": SingleIdTokenIndexer()}
        self._nonterminal_indexers = nonterminal_indexers or {u"tokens":
                                                              SingleIdTokenIndexer(u"rule_labels")}
        self._terminal_indexers = terminal_indexers or {u"tokens": SingleIdTokenIndexer(u"rule_labels")}
        self._output_agendas = output_agendas

    #overrides
    def _read(self, file_path     ):
        with open(file_path, u"r") as data_file:
            logger.info(u"Reading instances from lines in file: %s", file_path)
            for line in data_file:
                line = line.strip(u"\n")
                if not line:
                    continue
                data = json.loads(line)
                sentence = data[u"sentence"]
                identifier = data[u"identifier"] if u"identifier" in data else data[u"id"]
                if u"worlds" in data:
                    # This means that we are reading grouped nlvr data. There will be multiple
                    # worlds and corresponding labels per sentence.
                    labels = data[u"labels"]
                    structured_representations = data[u"worlds"]
                else:
                    # We will make lists of labels and structured representations, each with just
                    # one element for consistency.
                    labels = [data[u"label"]]
                    structured_representations = [data[u"structured_rep"]]

                target_sequences                  = None
                # TODO(pradeep): The processed file also has incorrect sequences as well, which are
                # needed if we want to define some sort of a hinge-loss based trainer. Deal with
                # them.
                if u"correct_sequences" in data:
                    # We are reading the processed file and these are the "correct" logical form
                    # sequences. See ``scripts/nlvr/get_nlvr_logical_forms.py``.
                    target_sequences = data[u"correct_sequences"]
                instance = self.text_to_instance(sentence,
                                                 structured_representations,
                                                 labels,
                                                 target_sequences,
                                                 identifier)
                if instance is not None:
                    yield instance

    #overrides
    def text_to_instance(self,  # type: ignore
                         sentence     ,
                         structured_representations                            ,
                         labels            = None,
                         target_sequences                  = None,
                         identifier      = None)            :
        u"""
        Parameters
        ----------
        sentence : ``str``
            The query sentence.
        structured_representations : ``List[List[List[JsonDict]]]``
            A list of Json representations of all the worlds. See expected format in this class' docstring.
        labels : ``List[str]`` (optional)
            List of string representations of the labels (true or false) corresponding to the
            ``structured_representations``. Not required while testing.
        target_sequences : ``List[List[str]]`` (optional)
            List of target action sequences for each element which lead to the correct denotation in
            worlds corresponding to the structured representations.
        identifier : ``str`` (optional)
            The identifier from the dataset if available.
        """
        # pylint: disable=arguments-differ
        worlds = [NlvrWorld(data) for data in structured_representations]
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        sentence_field = TextField(tokenized_sentence, self._sentence_token_indexers)
        production_rule_fields              = []
        instance_action_ids                 = {}
        # TODO(pradeep): Assuming that possible actions are the same in all worlds. This may change
        # later.
        for production_rule in worlds[0].all_possible_actions():
            instance_action_ids[production_rule] = len(instance_action_ids)
            field = ProductionRuleField(production_rule, is_global_rule=True)
            production_rule_fields.append(field)
        action_field = ListField(production_rule_fields)
        worlds_field = ListField([MetadataField(world) for world in worlds])
        fields                   = {u"sentence": sentence_field,
                                    u"worlds": worlds_field,
                                    u"actions": action_field}
        if identifier is not None:
            fields[u"identifier"] = MetadataField(identifier)
        # Depending on the type of supervision used for training the parser, we may want either
        # target action sequences or an agenda in our instance. We check if target sequences are
        # provided, and include them if they are. If not, we'll get an agenda for the sentence, and
        # include that in the instance.
        if target_sequences:
            action_sequence_fields              = []
            for target_sequence in target_sequences:
                index_fields = ListField([IndexField(instance_action_ids[action], action_field)
                                          for action in target_sequence])
                action_sequence_fields.append(index_fields)
                # TODO(pradeep): Define a max length for this field.
            fields[u"target_action_sequences"] = ListField(action_sequence_fields)
        elif self._output_agendas:
            # TODO(pradeep): Assuming every world gives the same agenda for a sentence. This is true
            # now, but may change later too.
            agenda = worlds[0].get_agenda_for_sentence(sentence, add_paths_to_agenda=False)
            assert agenda, u"No agenda found for sentence: %s" % sentence
            # agenda_field contains indices into actions.
            agenda_field = ListField([IndexField(instance_action_ids[action], action_field)
                                      for action in agenda])
            fields[u"agenda"] = agenda_field
        if labels:
            labels_field = ListField([LabelField(label, label_namespace=u'denotations')
                                      for label in labels])
            fields[u"labels"] = labels_field

        return Instance(fields)

NlvrDatasetReader = DatasetReader.register(u"nlvr")(NlvrDatasetReader)
