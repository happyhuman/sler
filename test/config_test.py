from unittest import TestCase
from config import ConfigManager


class TestConfigManager(TestCase):
    def setUp(self):
        pass

    def test_load_yaml(self):
        cm = ConfigManager()
        self.assertEqual(0, len(cm.estimators))
        cm.load_yaml('./yaml_sample.yml')
        self.assertEqual(1, len(cm.estimators))
        self.assertEqual('svc', cm.estimators[0]['type'])
        self.assertEqual({'degree': 4}, cm.estimators[0]['init'])
        self.assertEqual({'C', 'kernel'}, set(cm.estimators[0]['hyperparameters'].keys()))
        self.assertEqual({'rbf', 'linear'}, set(cm.estimators[0]['hyperparameters']['kernel']))
        self.assertEqual('all', cm.estimators[0]['generate'])

        self.assertEqual('sample.csv', cm.input_file)
        self.assertEqual('pred.csv', cm.prediction_file)
        self.assertEqual('report.txt', cm.report_file)
        self.assertEqual({'age', 'smoker', 'gender'}, set(cm.feature_names))
        self.assertEqual('cancer', cm.response_name)
        self.assertEqual('classification', cm.estimator_type)
        self.assertEqual(20, cm.test_percentage)
        self.assertEqual(['age'], cm.pre['standardize'])
        self.assertEqual({'age': 'mean'}, cm.pre['impute'])
