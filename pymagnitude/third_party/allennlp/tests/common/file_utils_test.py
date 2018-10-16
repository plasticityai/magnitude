# pylint: disable=no-self-use,invalid-name



from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from collections import Counter
import os
import pathlib
import json
import tempfile
#typing

import boto3
from moto import mock_s3
import pytest
import responses

from allennlp.common.file_utils import (
        url_to_filename, filename_to_url, get_from_cache, cached_path, split_s3_path,
        s3_request, s3_etag, s3_get)
from allennlp.common.testing import AllenNlpTestCase
from io import open


def set_up_glove(url     , byt       , change_etag_every      = 1000):
    # Mock response for the datastore url that returns glove vectors
    responses.add(
            responses.GET,
            url,
            body=byt,
            status=200,
            content_type=u'application/gzip',
            stream=True,
            headers={u'Content-Length': unicode(len(byt))}
    )

    etags_left = change_etag_every
    etag = u"0"
    def head_callback(_):
        u"""
        Writing this as a callback allows different responses to different HEAD requests.
        In our case, we're going to change the ETag header every `change_etag_every`
        requests, which will allow us to simulate having a new version of the file.
        """
        nonlocal etags_left, etag
        headers = {u"ETag": etag}
        # countdown and change ETag
        etags_left -= 1
        if etags_left <= 0:
            etags_left = change_etag_every
            etag = unicode(int(etag) + 1)
        return (200, headers, u"")

    responses.add_callback(
            responses.HEAD,
            url,
            callback=head_callback
    )


def set_up_s3_bucket(bucket_name      = u"my-bucket", s3_objects                        = None):
    u"""Creates a mock s3 bucket optionally with objects uploaded from local files."""
    s3_client = boto3.client(u"s3")
    s3_client.create_bucket(Bucket=bucket_name)
    for filename, key in s3_objects or []:
        s3_client.upload_file(Filename=filename, Bucket=bucket_name, Key=key)


class TestFileUtils(AllenNlpTestCase):
    def setUp(self):
        super(TestFileUtils, self).setUp()
        self.glove_file = self.FIXTURES_ROOT / u'embeddings/glove.6B.100d.sample.txt.gz'
        with open(self.glove_file, u'rb') as glove:
            self.glove_bytes = glove.read()

    def test_url_to_filename(self):
        for url in [u'http://allenai.org', u'http://allennlp.org',
                    u'https://www.google.com', u'http://pytorch.org',
                    u'https://s3-us-west-2.amazonaws.com/allennlp' + u'/long' * 20 + u'/url']:
            filename = url_to_filename(url)
            assert u"http" not in filename
            with pytest.raises(FileNotFoundError):
                filename_to_url(filename, cache_dir=self.TEST_DIR)
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            with pytest.raises(FileNotFoundError):
                filename_to_url(filename, cache_dir=self.TEST_DIR)
            json.dump({u'url': url, u'etag': None},
                      open(os.path.join(self.TEST_DIR, filename + u'.json'), u'w'))
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag is None

    def test_url_to_filename_with_etags(self):
        for url in [u'http://allenai.org', u'http://allennlp.org',
                    u'https://www.google.com', u'http://pytorch.org']:
            filename = url_to_filename(url, etag=u"mytag")
            assert u"http" not in filename
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            json.dump({u'url': url, u'etag': u'mytag'},
                      open(os.path.join(self.TEST_DIR, filename + u'.json'), u'w'))
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag == u"mytag"
        baseurl = u'http://allenai.org/'
        assert url_to_filename(baseurl + u'1') != url_to_filename(baseurl, etag=u'1')

    def test_url_to_filename_with_etags_eliminates_quotes(self):
        for url in [u'http://allenai.org', u'http://allennlp.org',
                    u'https://www.google.com', u'http://pytorch.org']:
            filename = url_to_filename(url, etag=u'"mytag"')
            assert u"http" not in filename
            pathlib.Path(os.path.join(self.TEST_DIR, filename)).touch()
            json.dump({u'url': url, u'etag': u'mytag'},
                      open(os.path.join(self.TEST_DIR, filename + u'.json'), u'w'))
            back_to_url, etag = filename_to_url(filename, cache_dir=self.TEST_DIR)
            assert back_to_url == url
            assert etag == u"mytag"

    def test_split_s3_path(self):
        # Test splitting good urls.
        assert split_s3_path(u"s3://my-bucket/subdir/file.txt") == (u"my-bucket", u"subdir/file.txt")
        assert split_s3_path(u"s3://my-bucket/file.txt") == (u"my-bucket", u"file.txt")

        # Test splitting bad urls.
        with pytest.raises(ValueError):
            split_s3_path(u"s3://")
            split_s3_path(u"s3://myfile.txt")
            split_s3_path(u"myfile.txt")

    @mock_s3
    def test_s3_bucket(self):
        u"""This just ensures the bucket gets set up correctly."""
        set_up_s3_bucket()
        s3_client = boto3.client(u"s3")
        buckets = s3_client.list_buckets()[u"Buckets"]
        assert len(buckets) == 1
        assert buckets[0][u"Name"] == u"my-bucket"

    @mock_s3
    def test_s3_request_wrapper(self):
        set_up_s3_bucket(s3_objects=[(unicode(self.glove_file), u"embeddings/glove.txt.gz")])
        s3_resource = boto3.resource(u"s3")

        @s3_request
        def get_file_info(url):
            bucket_name, s3_path = split_s3_path(url)
            return s3_resource.Object(bucket_name, s3_path).content_type

        # Good request, should work.
        assert get_file_info(u"s3://my-bucket/embeddings/glove.txt.gz") == u"text/plain"

        # File missing, should raise FileNotFoundError.
        with pytest.raises(FileNotFoundError):
            get_file_info(u"s3://my-bucket/missing_file.txt")

    @mock_s3
    def test_s3_etag(self):
        set_up_s3_bucket(s3_objects=[(unicode(self.glove_file), u"embeddings/glove.txt.gz")])
        # Ensure we can get the etag for an s3 object and that it looks as expected.
        etag = s3_etag(u"s3://my-bucket/embeddings/glove.txt.gz")
        assert isinstance(etag, unicode)
        assert etag.startswith(u"'") or etag.startswith(u'"')

        # Should raise FileNotFoundError if the file does not exist on the bucket.
        with pytest.raises(FileNotFoundError):
            s3_etag(u"s3://my-bucket/missing_file.txt")

    @mock_s3
    def test_s3_get(self):
        set_up_s3_bucket(s3_objects=[(unicode(self.glove_file), u"embeddings/glove.txt.gz")])

        with tempfile.NamedTemporaryFile() as temp_file:
            s3_get(u"s3://my-bucket/embeddings/glove.txt.gz", temp_file)
            assert os.stat(temp_file.name).st_size != 0

        # Should raise FileNotFoundError if the file does not exist on the bucket.
        with pytest.raises(FileNotFoundError):
            with tempfile.NamedTemporaryFile() as temp_file:
                s3_get(u"s3://my-bucket/missing_file.txt", temp_file)

    @responses.activate
    def test_get_from_cache(self):
        url = u'http://fake.datastore.com/glove.txt.gz'
        set_up_glove(url, self.glove_bytes, change_etag_every=2)

        filename = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename == os.path.join(self.TEST_DIR, url_to_filename(url, etag=u"0"))

        # We should have made one HEAD request and one GET request.
        method_counts = Counter(call.request.method for call in responses.calls)
        assert len(method_counts) == 2
        assert method_counts[u'HEAD'] == 1
        assert method_counts[u'GET'] == 1

        # And the cached file should have the correct contents
        with open(filename, u'rb') as cached_file:
            assert cached_file.read() == self.glove_bytes

        # A second call to `get_from_cache` should make another HEAD call
        # but not another GET call.
        filename2 = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename2 == filename

        method_counts = Counter(call.request.method for call in responses.calls)
        assert len(method_counts) == 2
        assert method_counts[u'HEAD'] == 2
        assert method_counts[u'GET'] == 1

        with open(filename2, u'rb') as cached_file:
            assert cached_file.read() == self.glove_bytes

        # A third call should have a different ETag and should force a new download,
        # which means another HEAD call and another GET call.
        filename3 = get_from_cache(url, cache_dir=self.TEST_DIR)
        assert filename3 == os.path.join(self.TEST_DIR, url_to_filename(url, etag=u"1"))

        method_counts = Counter(call.request.method for call in responses.calls)
        assert len(method_counts) == 2
        assert method_counts[u'HEAD'] == 3
        assert method_counts[u'GET'] == 2

        with open(filename3, u'rb') as cached_file:
            assert cached_file.read() == self.glove_bytes

    @responses.activate
    def test_cached_path(self):
        url = u'http://fake.datastore.com/glove.txt.gz'
        set_up_glove(url, self.glove_bytes)

        # non-existent file
        with pytest.raises(FileNotFoundError):
            filename = cached_path(self.FIXTURES_ROOT / u"does_not_exist" /
                                   u"fake_file.tar.gz")

        # unparsable URI
        with pytest.raises(ValueError):
            filename = cached_path(u"fakescheme://path/to/fake/file.tar.gz")

        # existing file as path
        assert cached_path(self.glove_file) == unicode(self.glove_file)

        # caches urls
        filename = cached_path(url, cache_dir=self.TEST_DIR)

        assert len(responses.calls) == 2
        assert filename == os.path.join(self.TEST_DIR, url_to_filename(url, etag=u"0"))

        with open(filename, u'rb') as cached_file:
            assert cached_file.read() == self.glove_bytes
