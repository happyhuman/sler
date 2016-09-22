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

# Example 
In order to run sler on the iris data (provided in scikit-learn), you can simply run:
```python
iris = datasets.load_iris()
run_sler(iris, 'iris.yml')
```

iris.yml is defined by:
```yaml
estimators:
  - estimator: logistic regression
  - estimator: svc
    parameters:
      degree: 4
      random_state: 7
    hyper parameters:
      C:
        - 1
        - 0.2
        - 0.003
      kernel:
        - rbf
        - linear
    generate: all
  - estimator: knn
    hyper parameters:
      n_neighbors:
        - 2
        - 3
        - 4
        - 5
        - 6
  - estimator: random forest
    parameters:
      n_estimators: 10
    hyper parameters:
      max_depth:
        - 3
        - 4
        - 5
        - 6
        - 7
    generate: random:3

pre:
  features:
    - 'sepal length (cm)'
    - 'sepal width (cm)'
    - 'petal length (cm)'
    - 'petal width (cm)'
  target: target
  rescale:
    - 'sepal length (cm)': standardize
    - 'sepal width (cm)': normalize
  impute:
    - 'petal length (cm)': mean
  split: 15 # %15 percent of the data will be used for testing.
```

By running the code above, you will get:

```
Preprocessing...
Training the estimators...
	training knn...
	training svc...
	training logistic regression...
	training random forest...
Creating predictions...
Accuracy Score for knn: 0.956522
Best hyper parameters for knn: {'n_neighbors': 3}

Accuracy Score for svc: 0.956522
Best hyper parameters for svc: {'kernel': 'linear', 'C': 1}

Accuracy Score for logistic regression: 0.869565
Best hyper parameters for logistic regression: {}

Accuracy Score for random forest: 0.913043
Best hyper parameters for random forest: {'max_depth': 7}

Accuracy Score for ensemble: 0.913043

   knn  svc  logistic regression  random forest  ensemble  actual
0    0    0                    0              0         0       0
1    1    1                    1              1         1       1
2    0    0                    0              0         0       0
3    0    0                    0              0         0       0
4    1    1                    1              1         1       1
5    0    0                    0              0         0       0
6    1    1                    2              1         1       1
7    1    1                    1              1         1       1
8    2    2                    2              2         2       2
9    2    2                    2              2         2       2
```

Further examples can be found in the test directory.