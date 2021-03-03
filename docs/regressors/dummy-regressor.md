<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/DummyRegressor.php">[source]</a></span>

# Dummy Regressor
A regressor that makes predictions based on a user-defined guessing strategy using only the information found within the labels of a training set. Dummy Regressor is useful to provide a sanity check and to compare performance against actual Regressors.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous, Resource

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | strategy | Mean | Strategy | The guessing strategy to employ when making predictions. |

## Example
```php
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Other\Strategies\Percentile;

$estimator = new DummyRegressor(new Percentile(56.5));
```

## Additional Methods
This estimator does not have any additional methods.
