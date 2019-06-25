<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Regressors/DummyRegressor.php">Source</a></span>

# Dummy Regressor
Regressor that guesses output values based on a user-defined guessing [Strategy](../other/strategies/api.md). Dummy Regressor is useful to provide a sanity check and to compare performance against actual Regressors.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous, Resource

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | strategy | Mean | object | The guessing strategy to employ when guessing the outcome of a sample. |

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\Regressors\DummyRegressor;
use Rubix\ML\Other\Strategies\BlurryPercentile;

$estimator = new DummyRegressor(new BlurryPercentile(56.5, 0.1));
```