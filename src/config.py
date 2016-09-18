import os.path
import logging


class ConfigManager(object):
    def __init__(self):
        self.hp = {}
        self.hp['svm.default'] = {'C': [0.01, 1, 100], 'kernel': ['rbf', 'linear']}
        self.input_file = None
        self.pre = None
        self.feature_names = None
        self.response_name = None
        self.estimator_type = None
        self.estimators = []
        self.regresison_estimators = {'linearregression', }
        self.classification_estimators = {'svm', 'ridge', 'knn'}
        self.clustering_estimators = {'kmeans'}

    def load_yaml(self, yaml_file):
        if os.path.exists(yaml_file):
            import yaml
            cfg_values = yaml.load(file(yaml_file, 'r'))
            self._process_config_values(cfg_values)
        else:
            logging.error("%s does not exist.", yaml_file)

    def _process_config_values(self, cfg):
        if 'input' in cfg:
            self._process_input_configuration(cfg['input'])
        else:
            logging.error("The configuration should specify the input")
            return
        if 'estimator' in cfg:
            if 'hyperparameters' in cfg:
                self._process_hp_configuration(cfg['hyperparameters'])
            self._process_estimator_configuration(cfg['estimator'])
        else:
            logging.error("The configuration should specify the estimator")

    def _process_estimator_configuration(self, estimator_cfg):
        logging.debug("Estimator config is %s", estimator_cfg)
        _type = estimator_cfg['type']
        if _type in self.regresison_estimators:
            self.estimator_type = 'regression'
        elif _type in self.classification_estimators:
            self.estimator_type = 'classification'
        elif _type in self.clustering_estimators:
            self.estimator_type = 'clustering'
        if self.estimator_type is None:
            logging.error("Unknown Estimator: '%s'", _type)
            return False
        if self.estimator_type in {'regression', 'classification'}:
            if self.response_name is None:
                logging.error("Response column is not specified")
                return False
        else:
            if self.response_name is not None:
                logging.warn("Response is not required for clustering. Ignoring it.")
        if 'hyperparameter' in estimator_cfg and (estimator_cfg['hyperparameter'] not in self.hp):
            default_hyperparameter = "%s.default"%_type
            if default_hyperparameter in self.hp:
                logging.warn("Hyperparameter '%s' is undefined. Using '%s.default' instead.", estimator_cfg['hyperparameter'], default_hyperparameter)
                estimator_cfg['hyperparameter'] = self.hp[default_hyperparameter]
            else:
                logging.error("Hyperparameter '%s' is undefined.", estimator_cfg['hyperparameter'])
                return False
        self.estimators.append(estimator_cfg)
        return True

    def _process_input_configuration(self, input_cfg):
        logging.debug("Input config is %s", input_cfg)
        if 'filename' in input_cfg:
            self.input_file = input_cfg['filename']
        else:
            logging.error("Input filename is not given.")
            return False
        if 'features' in input_cfg:
            self.feature_names = input_cfg['features']
        if 'response' in input_cfg:
            self.response_name = input_cfg['response']
        if 'pre' in input_cfg:
            self.pre = input_cfg['pre']
            if 'impute' in self.pre:
                for key, value in self.pre['impute'].iteritems():
                    if value not in {'mode', 'median', 'mean'}:
                        logging.warn("Impute value should be one of mode, median, mean. Unknown value: '%s'.", value)
        return True

    def _process_hp_configuration(self, hp_cfg):
        logging.debug("Hyperparameter config is %s", hp_cfg)
        self.hp[hp_cfg['name']] = hp_cfg['values']

