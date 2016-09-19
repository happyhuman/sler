from unittest import TestCase
from config import ConfigManager
from sler import run_sler
from sklearn import datasets


class TestSler(TestCase):
    def setUp(self):
        pass

    def test_load_yaml(self):
        cm = ConfigManager()
        self.assertEqual(0, len(cm.estimators))
        cm.load_yaml('./yaml_sample.yml')
        self.assertEqual(1, len(cm.estimators))
        self.assertEqual('svc', cm.estimators[0].name)
        self.assertEqual('classification', cm.estimators[0]._type)
        self.assertEqual({'degree': 4}, cm.estimators[0].parameters)
        self.assertEqual({'C', 'kernel'}, set(cm.estimators[0].hyperparameters.keys()))
        self.assertEqual({'rbf', 'linear'}, set(cm.estimators[0].hyperparameters['kernel']))
        self.assertEqual('all', cm.estimators[0].generate)

        self.assertEqual({'age', 'smoker', 'gender'}, set(cm.feature_names))
        self.assertEqual('cancer', cm.target_name)
        self.assertEqual([{'age': 'standardize'}], cm.rescale)
        self.assertEqual([{'age': 'mean'}], cm.impute)

    def test_sler_classification(self):
        iris = datasets.load_iris()
        run_sler(iris, 'iris.yml')

    def test_sler_regression(self):
        boston = datasets.load_boston()
        run_sler(boston, 'boston.yml')

