from sler import run_sler, ScikitLearnEasyRunner
from sklearn import datasets
import pandas as pd
import sys
import inspect
import sklearn.datasets


def example_iris(*params):
    iris = datasets.load_iris()
    run_sler(iris, 'iris.yml')


def example_boston(*params):
    boston = datasets.load_boston()
    run_sler(boston, 'boston.yml')


def example_titanic_yaml(*params):
    titanic_dataframe = pd.read_csv('titanic.csv')
    run_sler(titanic_dataframe, 'titanic.yml')


def example_titanic_json(*params):
    titanic_dataframe = pd.read_csv('titanic.csv')
    run_sler(titanic_dataframe, 'titanic.json')


def example_interactive(*params):
    sler = ScikitLearnEasyRunner('titanic.csv')
    _params = {'random_state': 1}
    _hyperparams = {'penalty': ('l1', 'l2'), 'C': (0.1, 1, 10)}
    sler.config.add_estimator('logistic regression', _params, _hyperparams)
    sler.config.set_target_name('Survived')
    sler.config.set_rescalings({'Fare': 'standardize', 'Age': 'normalize'})
    sler.config.set_imputations({'Age': 'mean'})
    sler.config.set_polynomial_features(['Age', 'Fare'], 3)
    sler.run()
    if len(params) > 0:
        print "Saving the model to '%s'"%params[0]
        import pickle
        best_model = sler.get_model('logistic regression')
        with open(params[0], 'wb') as handle:
            pickle.dump(best_model, handle)


def example_classification(*params):
    """
    :param params: number_features (default: 2), number_classes (default: 3), number_samples (default: 500)
    """
    n_features = int(params[0]) if len(params) > 0 else 2
    n_classes = int(params[1]) if len(params) > 1 else 3
    n_samples = int(params[2]) if len(params) > 2 else 500
    X, y = sklearn.datasets.make_classification(n_features=n_features, n_redundant=0, n_informative=2, \
                                                n_clusters_per_class=1, n_classes=n_classes, n_samples=n_samples)
    feature_names = ['feature_%d'%i for i in range(1, n_features+1)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    sler = ScikitLearnEasyRunner(df)
    _hyperparams = {'n_estimators': (10, 15, 20), 'max_depth': (3, 5, 7, 9), 'min_samples_split': (2, 3, 4, 5, 6, 7)}
    sler.config.add_estimator('random forest', None, _hyperparams, 'random:10')
    sler.config.set_target_name('target')
    sler.config.set_scorer('recall', {'average': 'weighted'})
    sler.run()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Please provide the example name that you want to run"
        sys.exit(1)
    example_name = 'example_' + sys.argv[1]
    all_functions = {name: data for name, data in inspect.getmembers(sys.modules[__name__], inspect.isfunction)}
    if example_name in all_functions:
        print "Running the example '%s'..."%sys.argv[1]
        all_functions[example_name](*sys.argv[2:])
    else:
        print "Unknown example name: %s" % sys.argv[1]
