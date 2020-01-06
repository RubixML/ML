<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Classifiers/NaiveBayes.php">[source]</a></span>

# Naive Bayes
Probability-based classifier that uses Bayes' Theorem and the strong assumption that all features are independent. In practice, the independence assumption tends to work out despite most features being correlated in the real world. This implementation is based on a multinomial (categorical) distribution of input features.

> **Note:** Each partial train has the overhead of recomputing the probability mass function for each feature per class. As such, it is better to train with fewer but larger training sets.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | alpha | 1.0 | float | The amount of additive (Laplace/Lidstone) smoothing applied to the probabilities. |
| 2 | priors | Auto | array | The class prior probabilities as an associative array with class labels as keys and the prior probabilities as values. |

## Additional Methods
Return the class prior probabilities:
```php
public priors() : array
```

Return the counts for each category per class:
```php
public counts() : array
```

## Example
```php
use Rubix\ML\Classifiers\NaiveBayes;

$estimator = new NaiveBayes(2.5, [
	'spam' => 0.3,
	'not spam' => 0.7,
]);
```