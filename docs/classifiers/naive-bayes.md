<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/NaiveBayes.php">Source</a></span>

# Naive Bayes
Probability-based classifier that estimates posterior probabilities of each class using Bayes' Theorem and the conditional probabilities calculated during training. The *naive* part relates to the fact that the algorithm assumes that all features are independent (non-correlated), which is not often the case in the real world but works well in practice.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | alpha | 1.0 | float | The amount of additive (Laplace/Lidstone) smoothing applied to the probabilities. |
| 2 | priors | Auto | array | The class prior probabilities as an associative array with class labels as keys and the prior probabilities as values. |

### Additional Methods
Return the class prior probabilities:
```php
public priors() : array
```

Return the counts for each category per class:
```php
public counts() : array
```

### Example
```php
use Rubix\ML\Classifiers\NaiveBayes;

$estimator = new NaiveBayes(2.5, [
	'spam' => 0.3,
	'not spam' => 0.7,
]);
```