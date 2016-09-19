from config import ConfigManager
import os.path
import pandas as pd
import logging
import sklearn
import sklearn.cross_validation
import sklearn.metrics
import argparse
import sklearn.preprocessing


class ScikitLearnEasyRunner(object):
    def __init__(self, _input, config_file, test_percentage=10):
        self.cfgm = ConfigManager()
        self.cfgm.load_yaml(config_file)
        self.test_percentage = test_percentage
        self.dataframe = None
        self.train_features = None
        self.train_target = None
        self.test_features = None
        self.test_target = None
        self.pred_df = None
        self.estimators = {e.name: e.get_estimator() for e in self.cfgm.estimators}
        self._load_input(_input)

    def _pre_process(self):
        self._rescale()
        self._impute()
        #TODO: Where should I do vectorization?
        self._prep_features_target()

    def _rescale(self):
        if self.cfgm.rescale is not None:
            for item in self.cfgm.rescale:
                feature = item.keys()[0]
                value = item[feature]
                if value == 'standardize':
                    self.dataframe[feature] = sklearn.preprocessing.StandardScaler().fit_transform(self.dataframe[feature].reshape(-1, 1))
                elif value in {'normalize', 'minmax'}:
                    self.dataframe[feature] = sklearn.preprocessing.MinMaxScaler().fit_transform(self.dataframe[feature].reshape(-1, 1))

    def _impute(self):
        if self.cfgm.impute is not None:
            for item in self.cfgm.impute:
                feature = item.keys()[0]
                value = item[feature]
                self._fillna(feature, value)

    def _load_input(self, _input):
        if isinstance(_input, str):
            if os.path.exists(_input):
                if _input.endswith('.csv'):
                    self.dataframe = pd.read_csv(_input)
                elif _input.endswith('.xlsx'):
                    self.dataframe = pd.read_excel(_input)
                else:
                    logging.error("Unable to read the input file. It has to be a csv or an xlsx file")
            else:
                logging.error("Input file '%s' does not exist")
        elif isinstance(_input, pd.DataFrame):
            self.dataframe = _input
        elif isinstance(_input, sklearn.datasets.base.Bunch):
            logging.info("Converting sklearn Bunch to pandas DataFrame...")
            self.dataframe = pd.DataFrame(_input.data, columns=_input['feature_names'])
            self.dataframe['target'] = _input.target
        else:
            logging.error("Unknown input type: %s", type(_input))

    def _prep_features_target(self):
        target = self.dataframe[self.cfgm.target_name].copy()
        if self.cfgm.feature_names is None:
            features = self.dataframe.copy()
            del features[self.cfgm.target_name]
        else:
            features = self.dataframe[self.cfgm.feature_names].copy()
        features = pd.get_dummies(features)
        splits = sklearn.cross_validation.train_test_split(features, target, test_size=self.test_percentage/100.0)
        self.train_features, self.test_features, self.train_target, self.test_target = splits

    def _fillna(self, feature, value):
        if value == 'mean':
            self.dataframe[feature].fillna(self.dataframe[feature].mean(), inplace=True)
        elif value == 'median':
            self.dataframe[feature].fillna(self.dataframe[feature].median(), inplace=True)
        elif value == 'mode':
            self.dataframe[feature].fillna(self.dataframe[feature].mode(), inplace=True)
        else:
            self.dataframe[feature].fillna(value, inplace=True)

    def fit(self):
        for name, est in self.estimators.iteritems():
            logging.error("Training estimator: %s", name)
            est.fit(self.train_features, self.train_target)

    def predict(self):
        self.pred_df = pd.DataFrame()
        self.pred_df['actual'] = self.test_target
        for _type, est in self.estimators.iteritems():
            self.pred_df[_type] = est.predict(self.test_features)
        if self.cfgm.estimator_type == 'regression':
            self.pred_df['ensemble (mean)'] = self.pred_df.mean(axis=1)
        elif self.cfgm.estimator_type == 'classification':
            self.pred_df['ensemble (mode)'] = self.pred_df.mean(axis=1)

    def report(self):
        for name in self.estimators:
            score = sklearn.metrics.accuracy_score(self.test_target, self.pred_df[name])
            print "Accuracy Score for %s: %f"%(name, score)
            if hasattr(self.estimators[name], 'best_params_'):
                print "Best hyper parameters for %s: %s"%(name, self.estimators[name].best_params_)
            print

    def run(self):
        self._pre_process()
        self.fit()
        self.predict()
        self.report()
        print self.pred_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help='input file, either a csv or an xlsx file')
    parser.add_argument("config_file", help='config file, given as a yaml')
    parser.add_argument("--testpercentage", help='percentage of rows used for testing. default is 10', type=int)

    args = parser.parse_args()
    output_file = args.output
    test_percentage = 10 if args.testpercentage is None else args.testpercentage
    sler = ScikitLearnEasyRunner(args.input_file, args.config_file, test_percentage)
    sler.run()


