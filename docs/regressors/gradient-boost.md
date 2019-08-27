<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/GradientBoost.php">Source</a></span>

# Gradient Boost
Gradient Boost is a stage-wise additive model that uses a Gradient Descent boosting method for training  boosters (Decision Trees) to correct the error residuals of a *weak* base learner. Stochastic gradient boosting is achieved by varying the ratio of samples to subsample uniformly at random from the training set.

> **Note:** The default base regressor is a Dummy Regressor using the Mean strategy and the default booster is a Regression Tree with a max depth of 3.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on base learners

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | booster | RegressionTree | object | The regressor that will fix up the error residuals of the *weak* base learner. |
| 2 | rate | 0.1 | float | The learning rate of the ensemble i.e. the *shrinkage* applied to each step. |
| 3 | estimators | 1000 | int | The number of estimators to train in the ensemble. |
| 4 | ratio | 0.5 | float | The ratio of samples to subsample from the training set for each booster. |
| 5 | min change | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 6 | window | 10 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 7 | holdout | 0.1 | float | The proportion of training samples to use for validation and progress monitoring. |
| 8 | metric | RSquared | object | The metric used to score the generalization performance of the model during training. |
| 9 | base | DummyRegressor | object | The *weak* base learner to be boosted. |

### Additional Methods
Return the normalized feature importances i.e. the proportion that each feature contributes to the overall model, indexed by feature column:
```php
public featureImportances() : array
```

Return the validation score at each epoch:
```php
public scores() : array
```

Return the training loss at each epoch:
```php
public steps() : array
```

### Example
```php
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\CrossValidation\Metrics\SMAPE;
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Other\Strategies\Constant;

$estimator = new GradientBoost(new RegressionTree(3), 0.1, 1000, 0.8, 1e-4, 15, 0.1, new SMAPE(), new DummyRegressor(new Constant(0.0)));
```

### References
>- J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
>- J. H. Friedman. (1999). Stochastic Gradient Boosting.
>- Y. Wei. et al. (2017). Early stopping for kernel boosting algorithms: A general analysis with localized complexities.