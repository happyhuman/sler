from sler import run_sler, ScikitLearnEasyRunner
from sklearn import datasets
import pandas as pd
import sys
import inspect


def iris():
    iris = datasets.load_iris()
    run_sler(iris, 'iris.yml')


def boston():
    boston = datasets.load_boston()
    run_sler(boston, 'boston.yml')


def titanic_yaml():
    titanic_dataframe = pd.read_csv('titanic.csv')
    run_sler(titanic_dataframe, 'titanic.yml')


def titanic_json():
    titanic_dataframe = pd.read_csv('titanic.csv')
    run_sler(titanic_dataframe, 'titanic.json')


def interactive():
    sler = ScikitLearnEasyRunner('titanic.csv')
    sler.config.add_estimator('logistic regression', {'random_state': 1}, {'penalty': ('l1', 'l2'), 'C': (0.1, 1, 10)})
    sler.config.set_target_name('Survived')
    sler.config.set_imputations({'Age': 'normalize'})
    sler.run()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Please provide the example name that you want to run"
        sys.exit(1)
    example_name = sys.argv[1]
    all_functions = {name: data for name, data in inspect.getmembers(sys.modules[__name__], inspect.isfunction)}
    if example_name in all_functions:
        print "Running the example '%s'..."%example_name
        all_functions[example_name]()
