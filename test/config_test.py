from unittest import TestCase
from config import ConfigManager


class TestPDUtil(TestCase):
    def setUp(self):
        pass

    def test_load_yaml(self):
        cm = ConfigManager()
        cm.load_yaml('./yaml_sample.yml')
        self.assertEqual(2, len(cm.hp))
        self.assertEqual(1, len(cm.estimators))
