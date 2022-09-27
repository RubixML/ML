<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/OneVsRest.php">[source]</a></span>

# One Vs Rest
One Vs Rest is an ensemble learner that trains a binary classifier to predict a particular class vs every other class for every possible class. The final class prediction is the class whose binary classifier returned the highest probability. One of the features of One Vs Rest is that it allows you to build a multiclass classifier out of an ensemble of otherwise binary classifiers.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Parallel](../parallel.md), [Persistable](../persistable.md)

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
