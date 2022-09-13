<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/NaiveBayes.php">[source]</a></span>

# Naive Bayes
Categorical Naive Bayes is a probability-based classifier that uses counting and Bayes' Theorem to derive the probabilities of a class given a sample of categorical features. The term *naive* refers to the fact that Naive Bayes treats each feature as if it was independent of the others even though this is usually not the case in real life.

!!! note
    Each partial train has the overhead of recomputing the probability mass function for each feature per class. As such, it is better to train with fewer but larger training sets.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | priors | null | array | The class prior probabilities as an associative array with class labels as keys and their prior probabilities as values totalling 1. If null, then priors will automatically be computed from the training data. |
| 2 | smoothing | 1.0 | float | The amount of Laplace smoothing added to the probabilities. |

## Example
```php
use Rubix\ML\Classifiers\NaiveBayes;

$estimator = new NaiveBayes([
	'spam' => 0.3,
	'not spam' => 0.7,
], 2.5);
```

## Additional Methods
Return the class prior probabilities:
```php
public priors() : float[]|null
```

Return the counts for each category per class:
```php
public counts() : array[]|null
```
