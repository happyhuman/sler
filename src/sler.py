from config import ConfigManager
import os.path
import pandas as pd
import logging
import sklearn
import sklearn.cross_validation
import sklearn.metrics
import argparse
import sklearn.preprocessing
import warnings


class ScikitLearnEasyRunner(object):
    def __init__(self, _input, config_file):
        warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
        self.cfgm = ConfigManager()
        self.cfgm.load_yaml(config_file)
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
        splits = sklearn.cross_validation.train_test_split(features, target, test_size=self.cfgm.test_percentage/100.0)
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
            logging.debug("Training estimator: %s", name)
            print "\ttraining %s..."%name
            est.fit(self.train_features, self.train_target)

    def predict(self):
        self.pred_df = pd.DataFrame()
        for _type, est in self.estimators.iteritems():
            self.pred_df[_type] = est.predict(self.test_features)
        if len(self.estimators) > 2:
            if self.cfgm.estimator_type == 'regression':
                self.pred_df['ensemble'] = self.pred_df.mean(axis=1)
            elif self.cfgm.estimator_type == 'classification':
                self.pred_df['ensemble'] = self.pred_df.mode(axis=1)
        self.pred_df['actual'] = self.test_target.values

    def _get_score(self, actual, prediction):
        if self.cfgm.estimator_type == 'classification':
            score = sklearn.metrics.accuracy_score(actual, prediction)
        else:
            score = sklearn.metrics.r2_score(actual, prediction)
        return score

    def report(self):
        for name in self.estimators:
            score = self._get_score(self.test_target, self.pred_df[name])
            print "Accuracy Score for %s: %f"%(name, score)
            if hasattr(self.estimators[name], 'best_params_'):
                print "Best hyper parameters for %s: %s"%(name, self.estimators[name].best_params_)
            print
        if 'ensemble' in self.pred_df:
            score = self._get_score(self.test_target, self.pred_df['ensemble'])
            print "Accuracy Score for ensemble: %f" % (score, )
            print
        print self.pred_df.head(10)
        print

    def run(self):
        print "Preprocessing..."
        self._pre_process()
        print "Training the estimators..."
        self.fit()
        print "Creating predictions..."
        self.predict()
        self.report()


def run_sler(_input, _config):
    """
    The function that runs sler engine, and prints a report
    :param _input: the input, which can be the path to a csv or xlsx file, a pandas DataFrame, or a scikit-learn Bunch
    :param _config: the yaml config file
    :return: the sler engine, for possible further processing
    """
    sler = ScikitLearnEasyRunner(_input, _config)
    sler.run()
    return sler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help='input file, either a csv or an xlsx file')
    parser.add_argument("config_file", help='config file, given as a yaml')

    args = parser.parse_args()
    run_sler(args.input_file, args.config_file)


