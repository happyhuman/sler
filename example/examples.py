from sler import run_sler, ScikitLearnEasyRunner
from sklearn import datasets
import pandas as pd
import sys
import inspect


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
    mparams = {'random_state': 1}
    mhyperparams = {'penalty': ('l1', 'l2'), 'C': (0.1, 1, 10)}
    sler.config.add_estimator('logistic regression', mparams, mhyperparams)
    sler.config.set_target_name('Survived')
    sler.config.set_imputations({'Age': 'normalize'})
    sler.run()
    if len(params) > 0:
        print "Saving the model to '%s'"%params[0]
        import pickle
        best_model = sler.get_model('logistic regression')
        with open(params[0], 'wb') as handle:
            pickle.dump(best_model, handle)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Please provide the example name that you want to run"
        sys.exit(1)
    example_name = 'example_' + sys.argv[1]
    all_functions = {name: data for name, data in inspect.getmembers(sys.modules[__name__], inspect.isfunction)}
    if example_name in all_functions:
        print "Running the example '%s'..."%sys.argv[1]
        all_functions[example_name](*sys.argv[2:])
