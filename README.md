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
- scipy (optional)

# Usage
In order to run sler, you need to define at least three elements:
- An input (e.g. a csv file, a scikit-learn Bunch, etc)
- The target/response column
- An estimator

However in most cases, this is not going to produce the desired model. There is usually a number of preprocessing steps that are required to prepare the input prior to model training. sler provides the following preprocessing capabilities:
- Filling the missing values by imputing, using the mean, median, or mode functions.
- Rescaling the numerical features, using standardization or normalization.
- Selecting a subet of available features for model training.
- Determining what percentage of the dataset should be allocated to testing.

sler automatically converts the categorical features to boolean features using One Hot Encoding. Hence, you need not do anything for this step.

You need to define at least one estimator to be able to run sler. However you may choose to define several estimators. All these estimators should be for either classification or regression. For every estimator, you can optionally define a list of parameters and hyper-parameters. Parameters and hyper-parameters are using to initialize an estimator. The following is an example in yaml:
```yaml
train:
    estimators:
      - estimator: svc
        parameters:
          degree: 4
        hyper parameters:
          C:
            - 1
            - 0.2
        generate: all
```

This tells sler to create, train, and evaluate the following two estimators:
- SVC(degree = 4, C = 1)
- SVC(degree = 4, C= 0.2)

sler will then train both of these estimators and evaluate them against the dataset to choose the best one. If there are many hyper-parameters, sler will have to create and train many estimators, which can be very time consuming. A more practical approach is to tell sler, to create only a subset of thse estimators at random and evaluate those. This can be controlled using the 'generate' parameter which is set to 'all' by default. You may set 'generate' to 'random:6' instead, to create only 6 estimators.

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
Analyzing the configuration...
Loading the input...
Pre-processing...
Training the estimators...
	training logistic regression...
Creating predictions...
accuracy score for logistic regression: 0.822222
	Best hyper parameters for logistic regression: {'penalty': 'l1', 'C': 1}

   logistic regression  actual
0                    1       1
1                    0       0
2                    0       0
3                    0       0
4                    0       1
5                    0       0
6                    0       1
7                    1       0
8                    1       1
9                    0       0
```