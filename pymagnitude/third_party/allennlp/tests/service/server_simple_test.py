# pylint: disable=no-self-use,invalid-name,line-too-long



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
import json
import os

import flask
import flask.testing

from allennlp.common.util import JsonDict
from allennlp.common.testing import AllenNlpTestCase
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.service.server_simple import make_app
from io import open


def post_json(client                           , endpoint     , data          )                  :
    return client.post(endpoint,
                       content_type=u"application/json",
                       data=json.dumps(data))

PAYLOAD = {
        u'passage': u"""The Matrix is a 1999 science fiction action film written and directed by The Wachowskis, starring Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Joe Pantoliano.""",
        u'question': u"""Who stars in the matrix?"""
}


class TestSimpleServer(AllenNlpTestCase):

    def setUp(self):
        super(TestSimpleServer, self).setUp()

        archive = load_archive(self.FIXTURES_ROOT / u'bidaf' / u'serialization' / u'model.tar.gz')
        self.bidaf_predictor = Predictor.from_archive(archive, u'machine-comprehension')


    def tearDown(self):
        super(TestSimpleServer, self).tearDown()
        try:
            os.remove(u'access.log')
            os.remove(u'error.log')
        except FileNotFoundError:
            pass

    def test_standard_model(self):
        app = make_app(predictor=self.bidaf_predictor, field_names=[u'passage', u'question'])
        app.testing = True
        client = app.test_client()

        # First test the HTML
        response = client.get(u'/')
        data = response.get_data()

        assert u"passage" in data
        assert u"question" in data

        # Now test the backend
        response = post_json(client, u'/predict', PAYLOAD)
        data = json.loads(response.get_data())
        assert u'best_span_str' in data
        assert u'span_start_logits' in data

    def test_sanitizer(self):
        def sanitize(result          )            :
            return dict((key, value) for key, value in list(result.items())
                    if key.startswith(u"best_span"))

        app = make_app(predictor=self.bidaf_predictor, field_names=[u'passage', u'question'], sanitizer=sanitize)
        app.testing = True
        client = app.test_client()

        response = post_json(client, u'/predict', PAYLOAD)
        data = json.loads(response.get_data())
        assert u'best_span_str' in data
        assert u'span_start_logits' not in data

    def test_static_dir(self):
        html = u"""<html><body>THIS IS A STATIC SITE</body></html>"""
        jpg = u"""something about a jpg"""

        with open(os.path.join(self.TEST_DIR, u'index.html'), u'w') as f:
            f.write(html)

        with open(os.path.join(self.TEST_DIR, u'jpg.txt'), u'w') as f:
            f.write(jpg)

        app = make_app(predictor=self.bidaf_predictor, static_dir=self.TEST_DIR)
        app.testing = True
        client = app.test_client()

        response = client.get(u'/')
        data = response.get_data().decode(u'utf-8')
        assert data == html

        response = client.get(u'jpg.txt')
        data = response.get_data().decode(u'utf-8')
        assert data == jpg
