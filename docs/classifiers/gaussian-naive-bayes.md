### Gaussian Naive Bayes
A variate of the [Naive Bayes](#naive-bayes) algorithm that uses a probability density function (*PDF*) over *continuous* features that are assumed to be normally distributed.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Classifiers/GaussianNB.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | priors | Auto | array | The user-defined class prior probabilities as an associative array with class labels as keys and the prior probabilities as values. |

**Additional Methods:**

Return the class prior probabilities:
```php
public priors() : array
```

Return the running mean of each feature column for each class:
```php
public means() : ?array
```

Return the running variance of each feature column for each class:
```php
public variances() : ?array
```

**Example:**

```php
use Rubix\ML\Classifiers\GaussianNB;

$estimator = new GaussianNB([
	'benign' => 0.9,
	'malignant' => 0.1,
]);
```

**References:**

>- T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.