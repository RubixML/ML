<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/GradientBoost.php">[source]</a></span>

# Gradient Boost
Gradient Boost (GBM) is a stage-wise additive ensemble that uses a Gradient Descent boosting scheme for training boosters (Decision Trees) to correct the error residuals of a series of *weak* base learners. The default base regressor is a [Dummy Regressor](dummy-regressor.md) using the [Mean](../strategies/mean.md) strategy and the default booster is a [Regression Tree](regression-tree.md) with a max height of 3.

!!! note
    Gradient Boost utilizes progress monitoring via an internal validation set for snapshotting and early stopping. If there are not enough training samples to build an internal validation set given the user-specified holdout ratio then training will proceed with progress monitoring disabled.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Verbose](../verbose.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on base learners

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | booster | RegressionTree | Learner | The regressor used to up the error residuals of the base learner. |
| 2 | rate | 0.1 | float | The learning rate of the ensemble i.e. the *shrinkage* applied to each step. |
| 3 | ratio | 0.5 | float | The ratio of samples to subsample from the training set to train each booster. |
| 4 | estimators | 1000 | int | The maximum number of boosters to train in the ensemble. |
| 5 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 6 | window | 10 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 7 | holdOut | 0.1 | float | The proportion of training samples to use for progress monitoring. |
| 8 | metric | RMSE | Metric | The metric used to score the generalization performance of the model during training. |
| 9 | base | DummyRegressor | Learner | The *weak* base learner to be boosted. |

## Example
```php
use Rubix\ML\Regressors\GradientBoost;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\CrossValidation\Metrics\SMAPE;
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Strategies\Constant;

$estimator = new GradientBoost(new RegressionTree(3), 0.1, 0.8, 1000, 1e-4, 10, 0.1, new SMAPE(), new DummyRegressor(new Constant(0.0)));
```

## Additional Methods
Return an iterable progress table with the steps from the last training session:
```php
public steps() : iterable
```

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->steps());
```

Return the validation score for each epoch from the last training session:
```php
public scores() : float[]|null
```

Return the loss for each epoch from the last training session:
```php
public losses() : float[]|null
```

## References
[^1]: J. H. Friedman. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
[^2]: J. H. Friedman. (1999). Stochastic Gradient Boosting.
[^3]: Y. Wei. et al. (2017). Early stopping for kernel boosting algorithms: A general analysis with localized complexities.
