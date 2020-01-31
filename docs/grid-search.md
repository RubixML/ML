<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/GridSearch.php">[source]</a></span>

# Grid Search
Grid Search is an algorithm that optimizes hyper-parameter selection. From the user's perspective, the process of training and predicting is the same, however, under the hood Grid Search trains a model for each combination of possible parameters and the best model is selected as the base estimator.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Parallel](parallel.md), [Persistable](persistable.md), [Verbose](verbose.md), [Wrapper](wrapper.md)

**Data Type Compatibility:** Depends on base learner

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | string | The class name of the base learner. |
| 2 | params | | array | An array of [tuples](faq.md#what-is-a-tuple) containing the possible values for each of the base learner's constructor parameters. |
| 3 | metric | Auto | Metric | The validation metric used to score each set of hyper-parameters. |
| 4 | validator | KFold | Validator | The validator used to test and score each trained model. |

## Additional Methods
Return an array of every possible combination of hyper-parameters:
```php
public combinations() : array
```

An [n-tuple](faq.md#what-is-a-tuple) containing the best parameters based on their validation score:
```php
public best() : array
```

Return the underlying base estimator:
```php
public estimator() : Estimator
```

## Example
```php
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\KFold;

$params = [
	[1, 3, 5, 10], [true, false], [new Euclidean(), new Manhattan()],
];

$estimator = new GridSearch(KNearestNeighbors::class, $params, new FBeta(), new KFold(5));

$estimator->train($dataset);

var_dump($estimator->best());
```

```sh
array(3) {
  [0]=> int(3)
  [1]=> bool(true)
  [2]=> object(Rubix\ML\Kernels\Distance\Manhattan) {}
}
```