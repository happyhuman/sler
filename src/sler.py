import os.path
import pandas as pd
import argparse
import warnings
import os.path
import logging
import sklearn
import sklearn.svm
import sklearn.metrics
import sklearn.ensemble
import sklearn.neighbors
import sklearn.grid_search
import sklearn.linear_model
import sklearn.datasets.base
import sklearn.preprocessing
import sklearn.cross_validation
import sklearn.metrics
import imp


def module_available(module_name):
    try:
        imp.find_module(module_name)
        found = True
    except ImportError:
        found = False
    return found


class EstimatorWrapper(object):
    _SVC_DEFAULT_H  = {}
    _SVC_DEFAULT_HP = {'C': [0.01, 1, 100], 'kernel': ['rbf', 'linear']}
    _REGRESSION_ESTIMATORS = {
        'linear regression': sklearn.linear_model.LinearRegression,
        'ridge': sklearn.linear_model.Ridge,
        'lasso': sklearn.linear_model.Lasso,
        'svr': sklearn.svm.SVR,
    }
    _CLASSIFICATION_ESTIMATORS = {
        'svc': sklearn.svm.SVC,
        'knn': sklearn.neighbors.KNeighborsClassifier,
        'random forest': sklearn.ensemble.RandomForestClassifier,
        'logistic regression': sklearn.linear_model.LogisticRegression,
        'gbc': sklearn.ensemble.GradientBoostingClassifier,
    }

    def __init__(self, name, parameters=None, hyperparameters=None, generate=None):
        self._type = None
        self.name = name
        self.hyperparameters = {} if hyperparameters is None else hyperparameters
        self.parameters = {} if parameters is None else parameters
        self.generate = 'all' if generate is None else generate
        if name in EstimatorWrapper._REGRESSION_ESTIMATORS:
            self._type = 'regression'
            if parameters is None:
                self.estimator = EstimatorWrapper._REGRESSION_ESTIMATORS[name]()
            else:
                self.estimator = EstimatorWrapper._REGRESSION_ESTIMATORS[name](**parameters)
        elif name in EstimatorWrapper._CLASSIFICATION_ESTIMATORS:
            self._type = 'classification'
            if parameters is None:
                self.estimator = EstimatorWrapper._CLASSIFICATION_ESTIMATORS[name]()
            else:
                self.estimator = EstimatorWrapper._CLASSIFICATION_ESTIMATORS[name](**parameters)
        else:
            logging.error("Unknown estimator: %s", name)

    def get_estimator(self):
        if self.generate == 'all':
            estimator = sklearn.grid_search.GridSearchCV(self.estimator, self.hyperparameters, n_jobs=2)
        elif self.generate.startswith('random'):
            n_iter = int(self.generate[len('random:'):]) if self.generate.startswith('random:') else 5
            estimator = sklearn.grid_search.RandomizedSearchCV(self.estimator, self.hyperparameters, n_iter, n_jobs=2)
        else:
            logging.error("Unknown generate parameter: %s. Ignoring hyperparameters", self.generate)
            estimator = self.estimator
        return estimator

    def __str__(self):
        return "EstimatorWrapper<%s>"%self.name

    @property
    def REGRESSION_ESTIMATORS(self):
        return self._REGRESSION_ESTIMATORS


class SlerConfigManager(object):
    _CLASSIFICATION_SCORERS = {
        'accuracy': sklearn.metrics.accuracy_score,
        'precision': sklearn.metrics.precision_score,
        'recall': sklearn.metrics.recall_score,
    }
    _REGRESSION_SCORERS = {
        'absolute': sklearn.metrics.mean_absolute_error,
        'squared': sklearn.metrics.mean_squared_error,
        'r2': sklearn.metrics.r2_score,
    }

    def __init__(self):
        self.feature_names = None
        self.target_name = None
        self.estimator_type = None
        self.rescale = None
        self.vectorize = None
        self.impute = None
        self.estimators = []
        self.test_percentage = 10
        self.scorer_name = None
        self.scorer_parameters = {}
        self.runnable = None

    def load_json(self, json_file):
        if os.path.exists(json_file):
            import json
            cfg_values = json.load(file(json_file, 'r'))
            self.runnable = self._process_config_values(cfg_values)
        else:
            logging.error("%s does not exist.", json_file)
            self.runnable = False

    def load_yaml(self, yaml_file):
        if module_available('yaml'):
            if os.path.exists(yaml_file):
                import yaml
                cfg_values = yaml.load(file(yaml_file, 'r'))
                self.runnable = self._process_config_values(cfg_values)
            else:
                logging.error("%s does not exist.", yaml_file)
        else:
            print "Please install pyyaml first"
            self.runnable = False

    def _assert(self, condition, message, level='error'):
        if not condition:
            if level == 'error':
                logging.error(message)
                self.runnable = False
            else:
                logging.warn(message)

    def analyze(self):
        self.runnable = True
        self._assert(self.target_name is not None, "Target name cannot be None")
        self._assert(len(self.estimators) > 0, "At least one estimator needs be defined")
        estimators_types = {self._get_estimator_type(est.name) for est in self.estimators}
        if len(estimators_types) == 1 and estimators_types.issubset({'classification', 'regression'}):
            self.estimator_type = list(estimators_types)[0]
        else:
            self._assert(False, "Not all estimators have the same type")
        if self.scorer_name is None:
            if self.estimator_type == 'classification':
                self.scorer_name = 'accuracy'
            else:
                self.scorer_name = 'r2'
        if self.scorer_name in SlerConfigManager._CLASSIFICATION_SCORERS:
            self._assert(self.estimator_type == 'classification', "Scoring type is different from the estimators type")
        elif self.scorer_name in SlerConfigManager._REGRESSION_SCORERS:
            self._assert(self.estimator_type == 'regression', "Scoring type is different from the estimators type")
        else:
            self._assert(False, "Unknown scoring method: %s" % self.scorer_name)
        if self.feature_names is not None:
            self._assert(self.target_name not in self.feature_names, "Target cannot also be one of the features")
        self._assert(0 < self.test_percentage < 100, "split should be between 1 and 99")


    def get_scorer(self):
        if self.scorer_name in SlerConfigManager._CLASSIFICATION_SCORERS:
            return SlerConfigManager._CLASSIFICATION_SCORERS[self.scorer_name]
        elif self.scorer_name in SlerConfigManager._REGRESSION_SCORERS:
            return SlerConfigManager._REGRESSION_SCORERS[self.scorer_name]
        return None

    def _process_config_values(self, cfg):
        if 'pre' in cfg:
            self._process_pre_configuration(cfg['pre'])
        else:
            logging.error("The configuration should have a 'pre' key")
            return False
        if 'train' in cfg:
            self._process_training_configuration(cfg['train'])
        else:
            logging.error("The configuration should have a 'train' key")
            return False
        return True

    def _process_training_configuration(self, training_cfg):
        if 'estimators' in training_cfg and len(training_cfg['estimators']) > 0:
            self._process_estimators_configuration(training_cfg['estimators'])
        else:
            logging.error("The configuration should specify at least one estimator")
        if 'scoring' in training_cfg:
            scoring = training_cfg['scoring']
            method = scoring['method']
            del scoring['method']
            self.set_scorer(method, scoring)

    def _process_estimators_configuration(self, estimators_cfg):
        logging.debug("Estimator config is %s", estimators_cfg)
        for est_cfg in estimators_cfg:
            self.add_estimator(est_cfg['estimator'], est_cfg.get('parameters'), est_cfg.get('hyper parameters'), est_cfg.get('generate'))
        return True

    def _process_pre_configuration(self, pre_cfg):
        logging.debug("Preprocess config is %s", pre_cfg)
        self.set_feature_names(pre_cfg.get('features'))
        self.set_target_name(pre_cfg.get('target'))
        self.set_imputations(pre_cfg.get('impute'))
        self.set_rescalings(pre_cfg.get('rescale'))
        self.vectorize = pre_cfg.get('vectorize')
        self.set_train_test_split(pre_cfg.get('split', 10))
        if self.target_name is None:
            logging.error("target should be specified")
        elif self.feature_names is not None and self.target_name in self.feature_names:
            logging.error("target cannot also be one of the features")

    def _get_estimator_type(self, name):
        if name in EstimatorWrapper._REGRESSION_ESTIMATORS:
            return 'regression'
        elif name in EstimatorWrapper._CLASSIFICATION_ESTIMATORS:
            return 'classification'
        return 'unknown'

    def remove_estimator(self, name):
        """
        Remove an estimator, given its name
        :param name: the name of the estimator to be removed
        """
        if name in self.estimators:
            del self.estimators[name]
            if len(self.estimators) == 0:
                self.estimator_type = None

    def add_estimator(self, name, parameters=None, hyperparameters=None, generate='all'):
        """
        Add a new estimator to be trained. If other estimators have been previously added, the type of the new estimator
        needs to match the type of the previous ones. If not other estimators had been added before, the type of this
        estimator sets the type for this configuration. A type is either a 'classification' or 'regression'.
        :param name: the estimator's name (e.g. svc, knn, ridge, etc)
        :param parameters: a dictionary of input names and values to be used to initialize the estimator
        :param hyperparameters: a dictionary to be used for tuning the estimator
        :param generate: either 'all', 'random', or 'random:N'. 'all' forces sler to create all possible models given
        the hyper parameters. 'random:N' only creates N many random models. If N is omitted, N will be set to 5.
        """
        est = EstimatorWrapper(name, parameters, hyperparameters, generate)
        self.estimators.append(est)

    def set_scorer(self, scorer_name, parameters = None):
        """
        Set the evaluator.
        :param name: For classifications, one of accuracy, precision, recall, or f1. For regression, one of absolute, squared, or r2.
        :param parameters: an optional dictionary of parameter names and values to be passed to the scorer
        """
        self.scorer_name = scorer_name
        self.scorer_parameters = {} if parameters is None else parameters

    def set_feature_names(self, names):
        """
        Choose which features to be used during training. If not set, all the existing features, except for target
        will be used. Setting feature names is optional.
        :param names: a list of of feature names
        """
        self.feature_names = names

    def set_target_name(self, name):
        """
        Set the target (response) column.
        :param name: the name of the target column.
        """
        self.target_name = name

    def set_imputations(self, imputations):
        """
        Define which features to be imputed and how.
        :param imputations: a dictionary of features names and imputation types. An imputation type should be 'mean',
        'median', 'mode', or a value.
        It is common to use mean or median for numerical features, and mode for categorical features.
        """
        self.impute = imputations

    def set_rescalings(self, rescalings):
        """
        Define how the numerical features should be rescaled and how.
        :param rescalings: a dictionary of feature names and rescaling type. A rescaling type is 'standardize'
        or 'normalize'.
        """
        self.rescale = rescalings

    def set_train_test_split(self, percentage):
        """
        Set what percentage of the data should be used for testing. Default is %10.
        :param percentage: the percentage used for testing.
        """
        self.test_percentage = percentage


class ScikitLearnEasyRunner(object):
    def __init__(self, _input, config_file=None):
        warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
        self._input = _input
        self.config = SlerConfigManager()
        self.dataframe = None
        self.original_dataframe = None
        self.train_features = None
        self.train_target = None
        self.test_features = None
        self.test_target = None
        self.pred_df = None
        self.estimators = None
        self.scorer = None
        if config_file is not None:
            if config_file.endswith('.json'):
                self.config.load_json(config_file)
            elif config_file.endswith('.yml') or config_file.endswith('.yaml'):
                self.config.load_yaml(config_file)
            else:
                logging.error("Unknown config extension: %s. Expecting a file with json, yml, or yaml extension", config_file)

    def _pre_process(self):
        self.scorer = self.config.get_scorer()
        self.estimators = {e.name: e.get_estimator() for e in self.config.estimators}
        self._rescale()
        self._impute()
        #TODO: Where should I do vectorization?
        self._prep_features_target()

    def _rescale(self):
        if self.config.rescale is not None:
            for feature, value in self.config.rescale.iteritems():
                if value == 'standardize':
                    self.dataframe[feature] = sklearn.preprocessing.StandardScaler().fit_transform(self.dataframe[feature].reshape(-1, 1))
                elif value in {'normalize', 'minmax'}:
                    self.dataframe[feature] = sklearn.preprocessing.MinMaxScaler().fit_transform(self.dataframe[feature].reshape(-1, 1))

    def _impute(self):
        if self.config.impute is not None:
            for feature, value in self.config.impute.iteritems():
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
        self.original_dataframe = self.dataframe.copy()

    def _prep_features_target(self):
        target = self.dataframe[self.config.target_name].copy()
        if self.config.feature_names is None:
            features = self.dataframe.copy()
            del features[self.config.target_name]
        else:
            features = self.dataframe[self.config.feature_names].copy()
        features = pd.get_dummies(features)
        splits = sklearn.cross_validation.train_test_split(features, target, test_size=self.config.test_percentage / 100.0)
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

    def reset(self):
        self.dataframe = self.original_dataframe.copy()

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
            if self.config.estimator_type == 'regression':
                self.pred_df['ensemble'] = self.pred_df.mean(axis=1)
            elif self.config.estimator_type == 'classification':
                if module_available('scipy'):
                    import scipy.stats
                    self.pred_df['ensemble'] = self.pred_df.apply(lambda x: scipy.stats.mode(x)[0][0], axis=1)
        self.pred_df['actual'] = self.test_target.values

    def _get_score(self, actual, prediction):
        if self.config.estimator_type == 'classification':
            self.scorer
            score = self.scorer(actual, prediction, **self.config.scorer_parameters)
        else:
            score = self.scorer(actual, prediction, **self.config.scorer_parameters)
        return score

    def get_model(self, name):
        if hasattr(self.estimators[name], 'best_estimator_'):
            return self.estimators[name].best_estimator_
        return self.estimators[name]

    def report(self):
        for name in self.estimators:
            score = self._get_score(self.test_target, self.pred_df[name])
            print "%s score for %s: %f"%(self.config.scorer_name, name, score)
            if hasattr(self.estimators[name], 'best_params_'):
                print "\tBest hyper parameters for %s: %s"%(name, self.estimators[name].best_params_)
            print
        if 'ensemble' in self.pred_df:
            score = self._get_score(self.test_target, self.pred_df['ensemble'])
            print "%s score for ensemble: %f" % (self.config.scorer_name, score, )
            print
        print self.pred_df.head(10)
        print

    def run(self):
        print "Analyzing the configuration..."
        self.config.analyze()
        if self.config.runnable:
            print "Loading the input..."
            self._load_input(self._input)
            print "Pre-processing..."
            self._pre_process()
            print "Training the estimators..."
            self.fit()
            print "Creating predictions..."
            self.predict()
            self.report()
        else:
            print "Cannot run sler due to configuration error"


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
    parser.add_argument("config_file", help='config file, given as a yaml or json file')

    args = parser.parse_args()
    run_sler(args.input_file, args.config_file)


