<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/BayesianSearch.php">[source]</a></span>

# Bayesian Search

Bayesian Search is a form of hyper-parameter optimization that uses the *Tree-structured Parzen Estimator* (TPE) to intelligently propose the next set of hyper-parameters to evaluate based on the results of the trials before it. Unlike [Grid Search](grid-search.md) which exhaustively evaluates every combination, Bayesian Search evaluates a sequence of trials one at a time, updating its beliefs after every trial, and with each new proposal concentrating its search on the regions of the hyper-parameter space that are most likely to yield an improvement. The process is sequential, so no [parallel](parallel.md) backend is used.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Persistable](persistable.md), [Verbose](verbose.md)

**Data Type Compatibility:** Depends on base learner

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | base | | string | The class name of the base learner. |
| 2 | params | | array | An array of lists containing the possible values for each of the base learner's constructor parameters. |
| 3 | metric | auto | Metric | The validation metric used to score each set of hyper-parameters. |
| 4 | validator | KFold | Validator | The validator used to test and score the model. |
| 5 | maxTrials | 32 | int | The maximum number of trials to run during a single training session. |
| 6 | quantile | 0.25 | float | The fraction of the top performing trials to use as the *good* set during sampling. |
| 7 | startup | 10 | int | The number of initial trials to run using random search before beginning estimation. |

## Example

```php
use Rubix\ML\BayesianSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\KFold;

$params = [
    [1, 3, 5, 10], [true, false], [new Euclidean(), new Manhattan()],
];

$estimator = new BayesianSearch(KNearestNeighbors::class, $params, new FBeta(), new KFold(5));

$estimator->train($dataset);

$estimator->predict($dataset);
```

Passing an empty array `[]` for any of the base learner's constructor parameters tells Bayesian Search to use that parameter's default value from the base learner's constructor (or `null` if no default exists).

You can also construct a Bayesian Search instance via the `fromNamedParams()` factory. Specify the hyper-parameters by the name of the base learner's constructor parameter (order does not matter). Hyper-parameters that are omitted are assigned their default value from the base learner's constructor.

```php
$estimator = BayesianSearch::fromNamedParams(
    KNearestNeighbors::class,
    [
        'kernel' => [new Euclidean(), new Manhattan()],
        'k' => [1, 3, 5, 10],
        'weighted' => [true, false],
    ],
    new FBeta(),
    new KFold(5)
);
```

## Additional Methods

Return the base learner instance.

```php
public base() : ?\Rubix\ML\Learner
```

Return an iterable table of every hyper-parameter combination tested along with its validation score from the last search, sorted by score descending.

```php
public results() : Generator
```

Return the best combination of parameters found during the last search along with their validation score in a 2-tuple.

```php
public best() : array
```

Return the validation scores of each of the trials in the order they were evaluated.

```php
public scores() : ?array
```

Return all the possible parameter combinations.

```php
public combinations() : array
```