# pytest: disable=no-self-use,invalid-name



from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
import shutil
import sys

import pytest

from allennlp.commands import main
from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase
from io import open

class TestMain(AllenNlpTestCase):
    def test_fails_on_unknown_command(self):
        sys.argv = [u"bogus",         # command
                    u"unknown_model", # model_name
                    u"bogus file",    # input_file
                    u"--output-file", u"bogus out file",
                    u"--silent"]

        with self.assertRaises(SystemExit) as cm:  # pylint: disable=invalid-name
            main()

        assert cm.exception.code == 2  # argparse code for incorrect usage

    def test_subcommand_overrides(self):
        def do_nothing(_):
            pass

        class FakeEvaluate(Subcommand):
            add_subparser_called = False

            def add_subparser(self, name, parser):
                subparser = parser.add_parser(name,
                                              description=u"fake",
                                              help=u"fake help")

                subparser.set_defaults(func=do_nothing)
                self.add_subparser_called = True

                return subparser

        fake_evaluate = FakeEvaluate()

        sys.argv = [u"allennlp.run", u"evaluate"]
        main(subcommand_overrides={u"evaluate": fake_evaluate})

        assert fake_evaluate.add_subparser_called

    def test_other_modules(self):
        # Create a new package in a temporary dir
        packagedir = self.TEST_DIR / u'testpackage'
        packagedir.mkdir()  # pylint: disable=no-member
        (packagedir / u'__init__.py').touch()  # pylint: disable=no-member

        # And add that directory to the path
        sys.path.insert(0, unicode(self.TEST_DIR))

        # Write out a duplicate model there, but registered under a different name.
        from allennlp.models import simple_tagger
        with open(simple_tagger.__file__) as model_file:
            code = model_file.read().replace(u"""@Model.register("simple_tagger")""",
                                             u"""@Model.register("duplicate-test-tagger")""")

        with open(packagedir / u'model.py', u'w') as new_model_file:
            new_model_file.write(code)

        # Copy fixture there too.
        shutil.copy(self.FIXTURES_ROOT / u'data' / u'sequence_tagging.tsv', self.TEST_DIR)
        data_path = unicode(self.TEST_DIR / u'sequence_tagging.tsv')

        # Write out config file
        config_path = self.TEST_DIR / u'config.json'
        config_json = u"""{
                "model": {
                        "type": "duplicate-test-tagger",
                        "text_field_embedder": {
                                "tokens": {
                                        "type": "embedding",
                                        "embedding_dim": 5
                                }
                        },
                        "encoder": {
                                "type": "lstm",
                                "input_size": 5,
                                "hidden_size": 7,
                                "num_layers": 2
                        }
                },
                "dataset_reader": {"type": "sequence_tagging"},
                "train_data_path": "$$$",
                "validation_data_path": "$$$",
                "iterator": {"type": "basic", "batch_size": 2},
                "trainer": {
                        "num_epochs": 2,
                        "optimizer": "adam"
                }
            }""".replace(u'$$$', data_path)
        with open(config_path, u'w') as config_file:
            config_file.write(config_json)

        serialization_dir = self.TEST_DIR / u'serialization'

        # Run train with using the non-allennlp module.
        sys.argv = [u"bin/allennlp",
                    u"train", unicode(config_path),
                    u"-s", unicode(serialization_dir)]

        # Shouldn't be able to find the model.
        with pytest.raises(ConfigurationError):
            main()

        # Now add the --include-package flag and it should work.
        # We also need to add --recover since the output directory already exists.
        sys.argv.extend([u"--recover", u"--include-package", u'testpackage'])

        main()

        # Rewrite out config file, but change a value.
        with open(config_path, u'w') as new_config_file:
            new_config_file.write(config_json.replace(u'"num_epochs": 2,', u'"num_epochs": 4,'))

        # This should fail because the config.json does not match that in the serialization directory.
        with pytest.raises(ConfigurationError):
            main()

        sys.path.remove(unicode(self.TEST_DIR))
