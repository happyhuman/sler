# sler (Scikit-Learn Easy Runner)

sler is a tool to simplify the usage of scikit-learn in many case.
There is usually a number of steps required to find the best estimator:
- rescale the numeric features through [standardization](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) or [normalization](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- fill in the missing values ([imputation](https://en.wikipedia.org/wiki/Imputation_(statistics)))
- select the features and the target
- encoding the categorical features using [One Hot encoding](https://en.wikipedia.org/wiki/One-hot)
- split the dataset for training and testing
- train one or more estimators using different techniques and hyper parameters
- evaluate the estimators by comparing their predictions against the test dataset
- choosing the best estimator using some scoring method

Using sler, you can perform most or all of the steps above through configuration, using a simple json or yaml file. You can define a number of estimators, along with the parameters and hyper-parameters, and let sler do all the work for you.
You can also define which features need be rescaled or imputed and how.

# Requirements
sler depends on the following libraries, which should be straight forward to install:
- numpy
- pandas
- scikit-learn
- pyyaml (optional, only needed if the config file is a yaml file)

# Examples
sler is designed to be easy to configure and run. There are several simple examples in the example directory to illustrate the basics of sler.
There are three ways to configure sler: using a yaml file, using a json file, or using the python API. The following simple example shows how to use sler directly using python:

```python
from sler import ScikitLearnEasyRunner
sler = ScikitLearnEasyRunner('titanic.csv')
params = {'random_state': 1}
hyperparams = {'penalty': ('l1', 'l2'), 'C': (0.1, 1, 10)}
sler.config.add_estimator('logistic regression', params, hyperparams) 
sler.config.set_target_name('Survived')
sler.config.set_imputations({'Age': 'mean'})
sler.run()
```

The following is the output of the example above:
```
Preprocessing...
Training the estimators...
	training logistic regression...
Creating predictions...
Accuracy Score for logistic regression: 0.722222
Best hyper parameters for logistic regression: {'penalty': 'l2', 'C': 10}

   logistic regression  actual
0                    0       0
1                    0       0
2                    0       0
3                    0       0
4                    1       0
5                    0       1
6                    0       0
7                    0       0
8                    0       0
9                    0       1
```