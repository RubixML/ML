<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/GridSearch.php">Source</a></span>

# Grid Search
Grid Search is an algorithm that optimizes hyper-parameter selection. From the user's perspective, the process of training and predicting is the same, however, under the hood, Grid Search trains one estimator per combination of parameters and the best model is selected as the base estimator.

> **Note:** You can choose the parameters to search manually or you can generate them randomly or in a grid using the [Params](other/helpers/params.md) helper.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Parallel](parallel.md), [Persistable](persistable.md), [Verbose](verbose.md)

**Data Type Compatibility:** Depends on base learner

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | string | The fully qualified class name of the base Estimator. |
| 2 | grid | | array | An array of [n-tuples](faq.md#what-is-a-tuple) where each tuple contains possible parameters for a given constructor location by ordinal. |
| 3 | metric | Auto | object | The validation metric used to score each set of hyper-parameters. Defaults to [F1 Score](cross-validation/metrics/f-beta.md) for classifiers and anomaly detectors, [R Squared](cross-validation/metrics/r-squared.md) for Regressors, and [V Measure](cross-validation/metrics/v-measure.md) for Clusterers. |
| 4 | validator | KFold | object | An instance of a validator object (HoldOut, KFold, etc.) that will be used to test each model. Defaults to K Fold with k of 5. |

### Additional Methods
Return an array of every possible parameter combination:
```php
public combinations() : array
```

A [tuple](faq.md#what-is-a-tuple) containing the best parameters and their validation score:
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
	[1, 3, 5, 10], [new Euclidean(), new Manhattan()], [true, false],
];

$estimator = new GridSearch(KNearestNeightbors::class, $grid, new FBeta(), new KFold(10));
```

```php
use Rubix\ML\GridSearch;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\CrossValidation\KFold;
use Rubix\CrossValidation\Metrics\VMeasure;

$params = [
	Params::grid(1, 5, 5), Params::floats(1.0, 20.0, 20), [new Diagonal(), new Minkowski(3.0)],
];

$estimator = new GridSearch(FuzzyCMeans::class, $params, new VMeasure(), new KFold(10));

$estimator->train($dataset);

var_dump($estimator->best());
```

**Output**

```sh
array(3) {
  [0]=> int(4)
  [1]=> float(13.65)
  [2]=> object(Rubix\ML\Kernels\Distance\Diagonal) {}
}
```