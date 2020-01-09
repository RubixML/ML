<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/BootstrapAggregator.php">[source]</a></span>

# Bootstrap Aggregator
Bootstrap Aggregating (or *bagging* for short) is a model averaging technique designed to improve the stability and performance of a user-specified base estimator by training a number of them on a unique *bootstrapped* training set sampled at random with replacement. Bagging works especially well with estimators that tend to have high variance by controlling the variance through averaging.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Parallel](parallel.md), [Persistable](persistable.md)

**Data Type Compatibility:** Depends on base learner

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | Learner | The base learner to be used in the ensemble. |
| 2 | estimators | 10 | int | The number of base estimators to train in the ensemble. |
| 3 | ratio | 0.5 | float | The ratio of samples (between 0 and 1.5) from the training set to train each base estimator with. |

## Additional Methods
This meta estimator does not have any additional methods.

## Example
```php
use Rubix\ML\BootstrapAggregator;
use Rubix\ML\Regressors\RegressionTree;

$estimator = new BootstrapAggregator(new RegressionTree(10), 300, 0.2);
```

### References
>- L. Breiman. (1996). Bagging Predictors.