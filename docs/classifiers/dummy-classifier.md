### Dummy Classifier
A classifier that uses a user-defined [Guessing Strategy](#guessing-strategies) to make predictions. Dummy Classifier is useful to provide a sanity check and to compare performance with an actual classifier.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/DummyClassifier.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Categorical, Continuous, Resource

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | strategy | Popularity Contest | object | The guessing strategy to employ when guessing the outcome of a sample. |

**Additional Methods:**

This estimator does not have any additional methods.

**Example:**

```php
use Rubix\ML\Classifiers\DummyClassifier;
use Rubix\ML\Other\Strategies\PopularityContest;

$estimator = new DummyClassifier(new PopularityContest());
```