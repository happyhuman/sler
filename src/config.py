import os.path
import logging
import sklearn.svm
import sklearn.neighbors
import sklearn.linear_model
import sklearn.grid_search
import sklearn.datasets.base
import sklearn.ensemble


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


class ConfigManager(object):
    def __init__(self):
        self.feature_names = None
        self.target_name = None
        self.estimator_type = None
        self.rescale = None
        self.vectorize = None
        self.impute = None
        self.estimators = []
        self.regression_estimators = {'linearregression', }
        self.classification_estimators = {'svc', 'ridge', 'knn'}
        self.test_percentage = None

    def load_json(self, json_file):
        if os.path.exists(json_file):
            import json
            cfg_values = json.load(file(json_file, 'r'))
            self._process_config_values(cfg_values)
        else:
            logging.error("%s does not exist.", json_file)

    def load_yaml(self, yaml_file):
        if os.path.exists(yaml_file):
            import yaml
            cfg_values = yaml.load(file(yaml_file, 'r'))
            self._process_config_values(cfg_values)
        else:
            logging.error("%s does not exist.", yaml_file)

    def _process_config_values(self, cfg):
        if 'pre' in cfg:
            self._process_pre_configuration(cfg['pre'])
        else:
            logging.error("The configuration should specify the input")
            return
        if 'estimators' in cfg:
            self._process_estimators_configuration(cfg['estimators'])
        else:
            logging.error("The configuration should specify at least one estimator")

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

    def _can_add_estimator(self, name):
        if len(self.estimators) > 0:
            return self._get_estimator_type(name) == self.estimator_type
        return True

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
        if self._can_add_estimator(name):
            est = EstimatorWrapper(name, parameters, hyperparameters, generate)
            self.estimators.append(est)
            self.estimator_type = est._type
        else:
            logging.warn("Cannot add %s. Its type does not match the existing estimators.")

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