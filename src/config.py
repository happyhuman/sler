import os.path
import logging


class ConfigManager(object):
    def __init__(self):
        self.default_hp = {}
        self.default_hp['svc.default'] = {'C': [0.01, 1, 100], 'kernel': ['rbf', 'linear']}
        self.input_file = None
        self.prediction_file = None
        self.report_file = None
        self.pre = None
        self.feature_names = None
        self.response_name = None
        self.estimator_type = None
        self.standardize = None
        self.vectorize = None
        self.estimators = []
        self.test_percentage = None
        self.regression_estimators = {'linearregression', }
        self.classification_estimators = {'svc', 'ridge', 'knn'}
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
        if 'output' in cfg:
            self._process_output_configuration(cfg['output'])
        else:
            logging.error("The configuration should specify the output")
            return
        if 'estimators' in cfg:
            self._process_estimators_configuration(cfg['estimators'])
        else:
            logging.error("The configuration should specify at least one estimator")

    def _process_estimators_configuration(self, estimators_cfg):
        logging.debug("Estimator config is %s", estimators_cfg)
        for estimator_cfg in estimators_cfg:
            _type = estimator_cfg['type']
            if _type in self.regression_estimators:
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
            self.estimators.append(estimator_cfg)
        return True

    def _process_output_configuration(self, output_cfg):
        if 'predictions' in output_cfg:
            self.prediction_file = output_cfg['predictions']
        else:
            logging.error("No file is specified to write the predictions output.")
            return False
        if 'report' in output_cfg:
            self.report_file = output_cfg['report']
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
        if 'standardize' in input_cfg:
            self.standardize = input_cfg['standardize']
        if 'vectorize' in input_cfg:
            self.vectorize = input_cfg['vectorize']
        if 'testpercentage' in input_cfg:
            self.test_percentage = input_cfg['testpercentage']
            if self.test_percentage < 1 or self.test_percentage > 99:
                logging.warn("testpercentage should be between 0 and 100. Setting it to 10")
                self.test_percentage = 10
        return True
