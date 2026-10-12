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
| 3 | initialTrials | 10 | int | The number of initial trials to run using random search before beginning estimation. |
| 4 | maxTrials | 32 | int | The maximum number of trials to run during a single training session. |
| 5 | quantile | 0.25 | float | The fraction of the top performing trials to use as the *good* set during sampling. |
| 6 | smoothing | 1.0 | float | The Laplace smoothing factor applied to the likelihood ratio between the *good* and *bad* candidate sets. A higher value produces a flatter, more uniform estimate (more exploration), while a lower value sharpens the preference for well-performing candidates (more exploitation). |
| 7 | metric | auto | Metric | The validation metric used to score each set of hyper-parameters. |
| 8 | validator | KFold | Validator | The validator used to test and score the model. |

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

$estimator = new BayesianSearch(KNearestNeighbors::class, $params, 10, 32, 0.25, 1.0, new FBeta(), new KFold(5));

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
    10,
    32,
    0.25,
    1.0,
    new FBeta(),
    new KFold(5)
);
```

## Setup

Some estimators expose configuration methods that are unrelated to their constructor parameters—for example, `setValidationDataset()` on iterative learners. Use `setup()` to register a callback that is invoked on each newly-instantiated base estimator before it is cross-validated (and again on the final best estimator before it is trained on the full dataset).

The callback receives the base estimator instance and may call any of its methods. It is not serialized with the bayesian search instance.

```php
$estimator = new BayesianSearch(LogisticRegression::class, $params, 10, 32, 0.25, 1.0, new FBeta(), new KFold(5));

$estimator->setup(function (LogisticRegression $regressor) : void {
    $regressor->setValidationDataset($validation);
});
```

## Additional Methods

Return the base learner instance.

```php
public base() : \Rubix\ML\Estimator
```

Return a Report of every hyper-parameter combination tested along with its validation score from the last search. The rows are keyed by trial number in the order the trials were evaluated, so the `Trial N` entries line up with the `N`-th combination (`scores()[N - 1]`) and the trial numbers logged during training. Each row pairs the validation score (keyed by the metric name) with a nested `params` map of the combination's constructor parameters:

```php
public results() : \Rubix\ML\Report
```

Return the validation scores of each of the trials in the order they were evaluated.

```php
public scores() : ?array
```

Return all the possible parameter combinations.

```php
public combinations() : array
```