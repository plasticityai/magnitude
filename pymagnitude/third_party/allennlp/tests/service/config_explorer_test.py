# pylint: disable=no-self-use,invalid-name,line-too-long,no-member




from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import json
import os
import pathlib
import sys

from allennlp.common.testing import AllenNlpTestCase
from allennlp.service import config_explorer
from allennlp.service.config_explorer import make_app, _HTML
from io import open


class TestConfigExplorer(AllenNlpTestCase):

    def setUp(self):
        super(TestConfigExplorer, self).setUp()
        app = make_app()
        app.testing = True
        self.client = app.test_client()

    def test_html(self):
        u"""
        The pip-installed version of allennlp (currently) requires the config explorer HTML
        to be hardcoded into the server file. But when iterating on it, it's easier to use the
        /debug/ endpoint, which points at `config_explorer.html`, so that you don't have to
        restart the server every time you make a change.

        This test just ensures that the two HTML versions are identical, to prevent you from
        making a change to the standalone HTML but forgetting to change the corresponding
        server HTML. There is certainly a better way to handle this.
        """
        config_explorer_dir = pathlib.Path(config_explorer.__file__).parent
        config_explorer_file = config_explorer_dir / u'config_explorer.html'

        if not config_explorer_file.exists():
            print(u"standalone config_explorer.html does not exist, skipping test")
        else:
            with open(config_explorer_file) as f:
                html = f.read()

            assert html.strip() == _HTML.strip()

    def test_app(self):
        response = self.client.get(u'/')
        html = response.get_data().decode(u'utf-8')

        assert u"AllenNLP Configuration Wizard" in html

    def test_api(self):
        response = self.client.get(u'/api/config/')
        data = json.loads(response.get_data())

        assert data[u"className"] == u""

        items = data[u"config"][u'items']

        assert items[0] == {
                u"name": u"dataset_reader",
                u"configurable": True,
                u"comment": u"specify your dataset reader here",
                u"annotation": {u'origin': u"allennlp.data.dataset_readers.dataset_reader.DatasetReader"}
        }


    def test_choices(self):
        response = self.client.get(u'/api/config/?class=allennlp.data.dataset_readers.dataset_reader.DatasetReader')
        data = json.loads(response.get_data())

        assert u"allennlp.data.dataset_readers.reading_comprehension.squad.SquadReader" in data[u"choices"]

    def test_subclass(self):
        response = self.client.get(u'/api/config/?class=allennlp.data.dataset_readers.semantic_role_labeling.SrlReader')
        data = json.loads(response.get_data())

        config = data[u'config']
        items = config[u'items']
        assert config[u'type'] == u'srl'
        assert items[0][u"name"] == u"token_indexers"

    def test_torch_class(self):
        response = self.client.get(u'/api/config/?class=torch.optim.rmsprop.RMSprop')
        data = json.loads(response.get_data())
        config = data[u'config']
        items = config[u'items']

        assert config[u"type"] == u"rmsprop"
        assert any(item[u"name"] == u"lr" for item in items)

    def test_rnn_hack(self):
        u"""
        Behind the scenes, when you try to create a torch RNN,
        it just calls torch.RNNBase with an extra parameter.
        This test is to make sure that works correctly.
        """
        response = self.client.get(u'/api/config/?class=torch.nn.modules.rnn.LSTM')
        data = json.loads(response.get_data())
        config = data[u'config']
        items = config[u'items']

        assert config[u"type"] == u"lstm"
        assert any(item[u"name"] == u"batch_first" for item in items)

    def test_initializers(self):
        response = self.client.get(u'/api/config/?class=allennlp.nn.initializers.Initializer')
        data = json.loads(response.get_data())

        assert u'torch.nn.init.constant_' in data[u"choices"]
        assert u'allennlp.nn.initializers.block_orthogonal' in data[u"choices"]

        response = self.client.get(u'/api/config/?class=torch.nn.init.uniform_')
        data = json.loads(response.get_data())
        config = data[u'config']
        items = config[u'items']

        assert config[u"type"] == u"uniform"
        assert any(item[u"name"] == u"a" for item in items)

    def test_regularizers(self):
        response = self.client.get(u'/api/config/?class=allennlp.nn.regularizers.regularizer.Regularizer')
        data = json.loads(response.get_data())

        assert u'allennlp.nn.regularizers.regularizers.L1Regularizer' in data[u"choices"]

        response = self.client.get(u'/api/config/?class=allennlp.nn.regularizers.regularizers.L1Regularizer')
        data = json.loads(response.get_data())
        config = data[u'config']
        items = config[u'items']

        assert config[u"type"] == u"l1"
        assert any(item[u"name"] == u"alpha" for item in items)

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / u'configexplorer'
        packagedir.mkdir()  # pylint: disable=no-member
        (packagedir / u'__init__.py').touch()  # pylint: disable=no-member

        # And add that directory to the path
        sys.path.insert(0, unicode(self.TEST_DIR))

        # Write out a duplicate predictor there, but registered under a different name.
        from allennlp.predictors import bidaf
        with open(bidaf.__file__) as f:
            code = f.read().replace(u"""@Predictor.register('machine-comprehension')""",
                                    u"""@Predictor.register('config-explorer-predictor')""")

        with open(os.path.join(packagedir, u'predictor.py'), u'w') as f:
            f.write(code)

        # Without specifying modules to load, it shouldn't be there
        app = make_app()
        app.testing = True
        client = app.test_client()
        response = client.get(u'/api/config/?class=allennlp.predictors.predictor.Predictor')
        data = json.loads(response.get_data())
        assert u"allennlp.predictors.bidaf.BidafPredictor" in data[u"choices"]
        assert u"configexplorer.predictor.BidafPredictor" not in data[u"choices"]

        # With specifying extra modules, it should be there.
        app = make_app([u'configexplorer'])
        app.testing = True
        client = app.test_client()
        response = client.get(u'/api/config/?class=allennlp.predictors.predictor.Predictor')
        data = json.loads(response.get_data())
        assert u"allennlp.predictors.bidaf.BidafPredictor" in data[u"choices"]
        assert u"configexplorer.predictor.BidafPredictor" in data[u"choices"]

        sys.path.remove(unicode(self.TEST_DIR))
