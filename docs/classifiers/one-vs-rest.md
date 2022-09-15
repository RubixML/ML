<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/OneVsRest.php">[source]</a></span>

# One Vs Rest
One Vs Rest is an ensemble learning technique that trains one binary classifier for each potential class label.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Parallel](../parallel.md)

**Data Type Compatibility:** Depends on the base learner

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | Learner|Probabilistic | The base classifier. |

## Example
```php
use Rubix\ML\Classifiers\OneVsRest;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Stochastic;

$estimator = new OneVsRest(new LogisticRegression(64, new Stochastic(0.001)));
```

## Additional Methods
This estimator does not have any additional methods.
