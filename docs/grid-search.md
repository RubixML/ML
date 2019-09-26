<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/GridSearch.php">Source</a></span>

# Grid Search
Grid Search is an algorithm that optimizes hyper-parameter selection. From the user's perspective, the process of training and predicting is the same, however, under the hood, Grid Search trains one estimator per combination of parameters and the best model is selected as the base estimator.

> **Note:** You can choose the hyper-parameters manually or you can generate them randomly or in a grid using the [Params](other/helpers/params.md) helper.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Parallel](parallel.md), [Persistable](persistable.md), [Verbose](verbose.md)

**Data Type Compatibility:** Depends on base learner

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | string | The fully qualified class name of the base Estimator. |
| 2 | grid | | array | An array of [tuples](faq.md#what-is-a-tuple) where each tuple contains the possible values of each parameter in the order they are given to the base learner's constructor. |
| 3 | metric | Auto | object | The validation metric used to score each set of hyper-parameters. |
| 4 | validator | KFold | object | An instance of a validator object (HoldOut, KFold, etc.) that will be used to test each model. |

> **Note:** The default validation metrics are [F Beta](cross-validation/metrics/f-beta.md) for classifiers and anomaly detectors, [R Squared](cross-validation/metrics/r-squared.md) for regressors, and [V Measure](cross-validation/metrics/v-measure.md) for clusterers.

### Additional Methods
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

### Example
```php
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\KFold;

$grid = [
	[1, 3, 5, 10], [true, false], [new Euclidean(), new Manhattan()],
];

$estimator = new GridSearch(KNearestNeightbors::class, $grid, new FBeta(), new KFold(5));

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