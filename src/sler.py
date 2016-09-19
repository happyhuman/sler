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
    def __init__(self, input_file, config_file, output_file=None, test_percentage=10):
        self.cfgm = ConfigManager()
        self.cfgm.load_yaml(config_file)
        self.input_file = input_file
        self.output_file=output_file
        self.test_percentage = test_percentage
        self.dataframe = None
        self.train_features = None
        self.train_response = None
        self.test_features = None
        self.test_response = None
        self.estimators = {e.name: e.get_estimator() for e in self.cfgm.estimators}

    def _pre_process(self):
        if self._load_input():
            self.rescale()
            self._impute()
            #TODO: Where should I do vectorization?
            self._prep_features_response()
            return True
        return False

    def rescale(self):
        if self.cfgm.rescale is not None:
            for item in self.cfgm.rescale:
                feature = item.keys()[0]
                value = item[feature]
                if value == 'standardize':
                    self.dataframe[feature] = sklearn.preprocessing.StandardScaler().fit_transform(self.dataframe[feature])
                elif value in {'normalize', 'minmax'}:
                    self.dataframe[feature] = sklearn.preprocessing.MinMaxScaler().fit_transform(self.dataframe[feature])

    def _impute(self):
        if self.cfgm.impute is not None:
            for item in self.cfgm.impute:
                feature = item.keys()[0]
                value = item[feature]
                self._fillna(feature, value)

    def _load_input(self):
        if os.path.exists(self.input_file):
            if self.input_file.endswith('.csv'):
                self.dataframe = pd.read_csv(self.input_file)
            elif self.input_file.endswith('.xlsx'):
                self.dataframe = pd.read_excel(self.input_file)
            else:
                logging.error("Unable to read the input file. It has to be a csv or an xlsx file")
                return False
        else:
            logging.error("Input file '%s' does not exist")
            return False
        return True

    def _prep_features_response(self):
        response = self.dataframe[self.cfgm.response_name].copy()
        if self.cfgm.feature_names is None:
            features = self.dataframe.copy()
            del features[self.cfgm.response_name]
        else:
            features = self.dataframe[self.cfgm.feature_names].copy()
        features = pd.get_dummies(features)
        self.train_features, self.test_features, self.train_response, self.test_response = \
            sklearn.cross_validation.train_test_split(features, response, test_size=self.test_percentage/100.0)

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
        for est in self.estimators.values():
            est.fit(self.train_features, self.train_response)

    def OLD_predict(self):
        self.pred_df = pd.DataFrame()
        for _type, est in self.estimators.iteritems():
            self.pred_df[_type] = est
        if self.cfgm.estimator_type == 'regression':
            self.pred_df['ensemble'] = self.pred_df.mean(axis=1)
        elif self.cfgm.estimator_type == 'classification':
            self.pred_df['ensemble'] = self.pred_df.mean(axis=1)
        self.pred_df['actual'] = self.test_response

    def report(self):
        print "Writing predictions to %s...."%self.cfgm.prediction_file
        self.pred_df.to_csv(self.cfgm.prediction_file)
        print "Writing report to %s...." % self.cfgm.report_file
        for name in self.estimators:
            score = sklearn.metrics.accuracy_score(self.pred_df['actual'], self.pred_df[name])
            print "Accuracy Score for %s: %f\n"%(name, score)
            if hasattr(self.estimators[name], 'best_params_'):
                print "Best parameters for %s: %s"%(name, self.estimators[name].best_params_)
            print

    def run(self):
        if self._pre_process():
            self.fit()
            #self.predict()
            self.report()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help='input file, either a csv or an xlsx file')
    parser.add_argument("config_file", help='config file, given as a yaml')
    parser.add_argument("--output", help='output file')
    parser.add_argument("--testpercentage", help='percentage of rows used for testing. default is 10', type=int)

    args = parser.parse_args()
    output_file = args.output
    test_percentage = 10 if args.testpercentage is None else args.testpercentage
    print args.input_file, args.config_file, output_file, test_percentage
    runner = ScikitLearnEasyRunner(args.input_file, args.config_file)
    runner.run()


