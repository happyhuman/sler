# sler (Scikit-Learn Easy Runner)

sler is a tool to simplify the usage of scikit-learn in many case.
There is usually a number of steps required to find the best estimator:
- rescale the numeric features through standardization or normalization
- fill in the missing values (imputation)
- select the features and the target
- split the dataset for training and testing
- train one or more estimators using different techniques and hyper parameters
- evaluate the estimators by comparing their predictions against the test dataset
- choosing the best estimator using some scoring method

Using sler, you can perform most or all of the steps above using a simple yaml file. You can define a number of estimators, along with the parameters and hyperparameters, and let sler do all the work for you.
You can also define which features need be rescaled or imputed and how.

# Example 
In order to run sler on the iris data (provided in scikit-learn), you can simply run:
```python
iris = datasets.load_iris()
sler = ScikitLearnEasyRunner(iris, 'iris.yml')
sler.run()
```

iris.yml is defined by:
```yaml
estimators:
  - type: svc
    parameters:
      degree: 4
      random_state: 7
    hyperparameters:
      C:
        - 1
        - 0.2
        - 0.003
      kernel:
        - rbf
        - linear
    generate: all
  - type: knn
    hyperparameters:
      n_neighbors:
        - 2
        - 3
        - 4
        - 5
        - 6
  - type: randomforest
    parameters:
      n_estimators: 10
    hyperparameters:
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
```

By running the code above, you will get:

```
Accuracy Score for knn: 0.933333
Best hyper parameters for knn: {'n_neighbors': 2}

Accuracy Score for randomforest: 0.933333
Best hyper parameters for randomforest: {'max_depth': 6}

Accuracy Score for svc: 0.933333
Best hyper parameters for svc: {'kernel': 'linear', 'C': 0.2}

Accuracy Score for ensemble: 0.933333

   knn  randomforest  svc  ensemble  actual
0    1             1    1         1       1
1    1             1    1         1       2
2    0             0    0         0       0
3    2             2    2         2       2
4    0             0    0         0       0
5    0             0    0         0       0
6    2             2    2         2       2
7    2             2    2         2       2
8    2             2    2         2       2
9    1             1    1         1       1
```
