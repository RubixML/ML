<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/GridSearch.php">[source]</a></span>

# Grid Search
Grid Search is an algorithm that optimizes hyper-parameter selection. From the user's perspective, the process of training and predicting is the same, however, under the hood Grid Search trains a model for each combination of possible parameters and the best model is selected as the base estimator.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Parallel](parallel.md), [Persistable](persistable.md), [Verbose](verbose.md)

**Data Type Compatibility:** Depends on base learner

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | base | | string | The class name of the base learner. |
| 2 | params | | array | An array of lists containing the possible values for each of the base learner's constructor parameters. |
| 3 | metric | auto | Metric | The validation metric used to score each set of hyper-parameters. |
| 4 | validator | KFold | Validator | The validator used to test and score the model. |

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
```

## Additional Methods
Return the base learner instance:
```php
public base() : ?\Rubix\ML\Learner
```
