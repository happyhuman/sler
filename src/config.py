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
            n_iter = int(self.generate[len('random:'):]) if self.generate.startswith('random:') else 10
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
        types = set()
        for est_cfg in estimators_cfg:
            est = EstimatorWrapper(est_cfg['estimator'], est_cfg.get('parameters'), est_cfg.get('hyperparameters'), est_cfg.get('generate'))
            types.add(est._type)
            self.estimators.append(est)
        if len(types) != 1:
            logging.error("All estimators should be either for classification or for regression, which is not currently the case.")
            return False
        else:
            self.estimator_type = list(types)[0]
        return True

    def _process_pre_configuration(self, pre_cfg):
        logging.debug("Preprocess config is %s", pre_cfg)
        self.feature_names = pre_cfg.get('features')
        self.target_name = pre_cfg.get('target')
        self.impute = pre_cfg.get('impute')
        self.rescale = pre_cfg.get('rescale')
        self.vectorize = pre_cfg.get('vectorize')
        self.test_percentage = pre_cfg.get('split', 10)
        if self.target_name is None:
            logging.error("target should be specified")
        elif self.feature_names is not None and self.target_name in self.feature_names:
            logging.error("target cannot also be one of the features")
