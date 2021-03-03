<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/GaussianNB.php">[source]</a></span>

# Gaussian Naive Bayes
Gaussian Naive Bayes is a version of the [Naive Bayes](naive-bayes.md) classifier for continuous features. It places a probability density function (PDF) over the features conditioned on a class basis and uses Bayes' Theorem to derive the final probabilities. In addition to the naive feature independence assumption, Gaussian Naive Bayes also assumes that all features are normally (Gaussian) distributed.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | priors | null | array | The class prior probabilities as an associative array with class labels as keys and the prior probabilities as values. If null, then the learner will compute these values from the training data. |

## Example
```php
use Rubix\ML\Classifiers\GaussianNB;

$estimator = new GaussianNB([
	'benign' => 0.9,
	'malignant' => 0.1,
]);
```

## Additional Methods
Return the class prior probabilities:
```php
public priors() : float[]|null
```

Return the running mean of each feature column for each class:
```php
public means() : array[]|null
```

Return the running variance of each feature column for each class:
```php
public variances() : array[]|null
```

## References
[^1]: T. F. Chan et al. (1979). Updating Formulae and a Pairwise Algorithm for Computing Sample Variances.