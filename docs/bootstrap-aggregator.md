<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/BootstrapAggregator.php">[source]</a></span>

# Bootstrap Aggregator
Bootstrap Aggregating (or *bagging* for short) is a model averaging technique designed to improve the stability and performance of a user-specified base estimator by training a number of them on a unique *bootstrapped* training set sampled at random with replacement. Bagging works especially well with estimators that tend to have high prediction variance by reducing the variance through averaging.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Parallel](parallel.md), [Persistable](persistable.md)

**Data Type Compatibility:** Depends on base learner

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | Learner | The base learner. |
| 2 | estimators | 10 | int | The number of base learners to train in the ensemble. |
| 3 | ratio | 0.5 | float | The ratio of samples from the training set to randomly subsample to train each base learner. |

## Example
```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

$estimator = new BootstrapAggregator(new RegressionTree(10), 300, 0.2);
```

## Additional Methods
This meta estimator does not have any additional methods.

## References
[^1]: L. Breiman. (1996). Bagging Predictors.