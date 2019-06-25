<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/DummyClassifier.php">Source</a></span>

# Dummy Classifier
A classifier that uses a user-defined guessing [Strategy](../other/strategies/api.md) to make predictions. Dummy Classifier is useful to provide a sanity check and to compare performance with an actual classifier.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous, Resource

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | strategy | Popularity Contest | object | The guessing strategy to employ when guessing the outcome of a sample. |

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Other\Strategies\PopularityContest;

$estimator = new DummyClassifier(new PopularityContest());
```