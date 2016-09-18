import sys
from config import ConfigManager
import os.path
import pandas as pd
import logging
import sklearn
import sklearn.linear_model
import sklearn.grid_search
import sklearn.neighbors


class ScikitLearnEasyRunner(object):
    def __init__(self, config_manager):
        self.cfgm = config_manager
        self.dataframe = None
        self.features = None
        self.response = None
        self.estimators = {}


    def _pre_process(self):
        if os.path.exists(self.cfgm.input_file):
            if self.cfgm.input_file.endswith('.csv'):
                self.dataframe = pd.read_csv(self.cfgm.input_file)
            elif self.cfgm.input_file.endswith('.xlsx'):
                self.dataframe = pd.read_excel(self.cfgm.input_file)
            else:
                logging.error("Unable to read the input file. It has to be a csv or an xlsx file")
                return False
            if self.cfgm.pre is not None:
                if 'impute' in self.cfgm.pre:
                    for column, value in self.cfgm.pre['impute']:
                        self._fillna(column, value)
            self._prep_features_response()
        else:
            logging.error("Input file '%s' does not exist")
            return False
        return True

    def _prep_features_response(self):
        if self.cfgm.feature_names is None:
            self.features = self.dataframe.copy()
            if self.cfgm.response_name is not None:
                del self.features[self.cfgm.response_name]
        else:
            self.features = self.dataframe[self.cfgm.feature_names].copy()
        self.features = pd.get_dummies(self.features)
        self.response = self.dataframe[self.cfgm.response_name].copy()

    def _fillna(self, column, value):
        if value == 'mean':
            self.dataframe[column].fillna(self.dataframe[column].mean(), inplace=True)
        elif value == 'median':
            self.dataframe[column].fillna(self.dataframe[column].median(), inplace=True)
        elif value == 'mode':
            self.dataframe[column].fillna(self.dataframe[column].mode(), inplace=True)
        else:
            self.dataframe[column].fillna(value, inplace=True)

    def _create_estimators(self):
        for est in self.cfgm.estimators:
            _type = est['type']
            init_params = est['init'] if 'init' in est else {}
            estimator = self._get_estimator(_type, init_params)
            gen = est['generate']
            param_grid = self.cfgm.hp[est['hyperparameter']]
            if gen == 'all':
                searchCV = sklearn.grid_search.GridSearchCV(estimator, param_grid, n_jobs=2)
            elif gen.startswith('random'):
                n_iter = int(gen[len('random'):]) if gen.startswith('random:') else 10
                searchCV = sklearn.grid_search.RandomizedSearchCV(estimator, param_grid, n_iter, n_jobs=2)
            self.estimators[_type] = searchCV

    def _get_estimator(self, _type, init_params):
        estimator = None
        if _type == 'svc':
            estimator = sklearn.svm.SVC(**init_params)
        elif _type == 'linearregression':
            estimator = sklearn.linear_model.LinearRegression(**init_params)
        elif _type == 'knn':
            estimator = sklearn.neighbors.NearestNeighbors(**init_params)
        if estimator is None:
            logging.error("Unknown estimator type: %s", _type)
        return estimator

    def fit(self):
        pass

    def run(self):
        if self._pre_process():
            self._create_estimators()
            if len(self.estimators) > 1:
                self.dataframe = pd.get_dummies(self.dataframe)


if __name__ == '__main__':
    args = sys.argv
    if len(args) < 2:
        print "Please specify a config file"
        sys.exit(1)
    cfgm = ConfigManager()
    cfgm.load_yaml(args[1])
    runner = ScikitLearnEasyRunner(cfgm)
    runner.run()

    print "Done"


