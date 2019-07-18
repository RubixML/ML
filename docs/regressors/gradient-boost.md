<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/GradientBoost.php">Source</a></span>

# Gradient Boost
Gradient Boost is a stage-wise additive model that uses a Gradient Descent boosting paradigm for training  boosters (Decision Trees) to correct the error residuals of a *weak* base learner.

> **Note:** The default base regressor is a Dummy Regressor using the *Mean* Strategy and the default booster is a Regression Tree with a max depth of 3.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on base learners

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | booster | RegressionTree | object | The regressor that will fix up the error residuals of the base learner. |
| 2 | rate | 0.1 | float | The learning rate of the ensemble i.e. the *shrinkage* parameter. |
| 3 | estimators | 1000 | int | The number of estimators to train in the ensemble. |
| 4 | ratio | 0.5 | float | The ratio of samples to subsample from the training dataset per epoch. |
| 5 | min change | 1e-4 | float | The minimum change in the cost function necessary to continue training. |
| 6 | holdout | 0.1 | float | The ratio of samples to hold out for progress monitoring. |
| 7 | window | 3 | int | The number of epochs to consider when determining an early stop. |
| 8 | metric | RSquared | object | The metric used to score the generalization performance of the model during training. |
| 9 | base | DummyRegressor | object | The *weak* learner to be boosted. |

### Additional Methods
Return the training error at each epoch:
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

$estimator = new GradientBoost(new RegressionTree(3), 0.1, 1000, 0.5, 1e-4, 0.1, 6, new SMAPE(), new DummyRegressor(new Constant(0.0)));
```

### References
>- J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
>- J. H. Friedman. (1999). Stochastic Gradient Boosting.