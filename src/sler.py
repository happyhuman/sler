import sys
from config import ConfigManager
import os.path
import pandas as pd
import logging
import sklearn
import sklearn.linear_model
import sklearn.grid_search
import sklearn.neighbors
import sklearn.cross_validation
import sklearn.metrics.accuracy_score


class ScikitLearnEasyRunner(object):
    def __init__(self, config_manager):
        self.cfgm = config_manager
        self.dataframe = None
        self.train_features = None
        self.train_response = None
        self.test_features = None
        self.test_response = None
        self.pred_df = None
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
        if self.cfgm.response_name is not None:
            response = self.dataframe[self.cfgm.response_name].copy()
        if self.cfgm.feature_names is None:
            features = self.dataframe.copy()
            del features[self.cfgm.response_name]
        else:
            features = self.dataframe[self.cfgm.feature_names].copy()

        features = pd.get_dummies(features)

        self.train_features, self.test_features, self.train_response, self.test_response = \
            sklearn.cross_validation.train_test_split(features, response, test_size=self.cfgm/100.0)
        self.train_features = pd.get_dummies(self.train_features)
        self.train_response = self.dataframe[self.cfgm.response_name].copy()

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
            param_grid = {}
            if 'hyperparameter' in est:
                param_grid = est['hyperparameter']
            elif "%s.default"%_type in self.cfgm.hp:
                param_grid = self.cfgm.hp["%s.default"%_type]
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
        for est in self.estimators.values():
            est.fit(self.train_features, self.train_response)

    def predict(self):
        self.pred_df = pd.DataFrame()
        for _type, est in self.estimators.iteritems():
            self.pred_df[_type] = est
        if self.cfgm.estimator_type == 'regression':
            self.pred_df['ensemble'] = self.pred_df.mean(axis=1)
        elif self.cfgm.estimator_type == 'classification':
            self.pred_df['ensemble'] = self.pred_df.mean(axis=1)
        self.pred_df['actual'] = self.test_response

    def write(self):
        print "Writing predictions to %s...."%self.cfgm.prediction_file
        self.pred_df.to_csv(self.cfgm.prediction_file)
        print "Writing report to %s...." % self.cfgm.report_file
        rep_file = file(self.cfgm.report_file, 'wt')
        for _type in self.estimators:
            score = sklearn.metrics.accuracy_score(self.pred_df['actual'], self.pred_df[_type])
            rep_file.write("Accuracy Score for %s: %f\n"%(_type, score))
        rep_file.close()
        print "Finished."

    def run(self):
        if self._pre_process():
            self._create_estimators()
            self.fit()
            self.predict()
            self.write()


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


