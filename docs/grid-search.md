<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/GridSearch.php">Source</a></span></p>

# Grid Search
Grid Search is an algorithm that optimizes hyper-parameter selection. From the user's perspective, the process of training and predicting is the same, however, under the hood, Grid Search trains one estimator per combination of parameters and the best model is selected as the base estimator.

> **Note**: You can choose the parameters to search manually or you can generate them randomly or in a grid using the [Params](#params) helper.

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Parallel](#parallel), [Persistable](#persistable), [Verbose](#verbose)

**Data Type Compatibility:** Depends on the base learner

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | string | The fully qualified class name of the base Estimator. |
| 2 | grid | | array | An array of [n-tuples](#what-is-a-tuple) where each tuple contains possible parameters for a given constructor location by ordinal. |
| 3 | metric | Auto | object | The validation metric used to score each set of hyper-parameters. |
| 4 | validator | KFold | object | An instance of a validator object (HoldOut, KFold, etc.) that will be used to test each model. |

### Additional Methods
Return an array of every possible parameter combination:
```php
public combinations() : array
```

A [tuple](#what-is-a-tuple) containing the best parameters and their validation score:
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