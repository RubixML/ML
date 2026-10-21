<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/GeneticSearch.php">[source]</a></span>

# Genetic Search

Genetic Search is a hyper-parameter optimizer that evolves a population of candidate parameter combinations. It scores candidates by cross-validation, selects parents by tournament, combines their parameter values with uniform crossover, and occasionally replaces a value through mutation. The best candidates survive between generations through elitism. Like [Bayesian Search](bayesian-search.md), Genetic Search evaluates a bounded subset of the search space rather than every combination. Its candidate values must be supplied as finite lists; it does not generate values outside those lists.

**Interfaces:** [Estimator](estimator.md), [Learner](learner.md), [Persistable](persistable.md), [Verbose](verbose.md)

**Data Type Compatibility:** Depends on base learner

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | class | | string | The class name of the base learner. |
| 2 | params | | array | An array of finite lists containing the possible values for each of the base learner's constructor parameters. |
| 3 | populationSize | 16 | int | The number of candidates in each generation. |
| 4 | generations | 10 | int | The maximum number of generations to evolve. |
| 5 | crossoverRate | 0.8 | float | The probability of combining two parents to produce an offspring. |
| 6 | mutationRate | 0.1 | float | The probability of replacing each offspring parameter with a randomly chosen candidate value. |
| 7 | eliteCount | 1 | int | The number of top-performing candidates preserved for the next generation. |
| 8 | metric | auto | Metric | The validation metric used to score each candidate. |
| 9 | validator | KFold | Validator | The validator used to test and score the model. |

## Example

```php
use Rubix\ML\GeneticSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\KFold;

$params = [
    [1, 3, 5, 10], [true, false], [new Euclidean(), new Manhattan()],
];

$estimator = new GeneticSearch(KNearestNeighbors::class, $params, 16, 10, 0.8, 0.1, 1, new FBeta(), new KFold(5));

$estimator->train($dataset);

$estimator->predict($dataset);
```

Each list corresponds to one base learner constructor parameter in constructor order. Passing an empty array `[]` for a parameter uses the base learner's constructor default (or `null` if no default exists). Genetic Search evaluates each distinct parameter combination at most once per training session. It evaluates no more than `populationSize * generations` unique combinations and stops when the finite search space is exhausted. After the search, it trains a new base learner with the best hyper-parameters on the full dataset.

You can also construct Genetic Search with `fromNamedParams()`. Specify hyper-parameters by the names of the base learner's constructor parameters, in any order. Omitted parameters use their constructor defaults.

```php
$estimator = GeneticSearch::fromNamedParams(
    KNearestNeighbors::class,
    [
        'kernel' => [new Euclidean(), new Manhattan()],
        'k' => [1, 3, 5, 10],
        'weighted' => [true, false],
    ],
    16,
    10,
    0.8,
    0.1,
    1,
    new FBeta(),
    new KFold(5)
);
```

## Setup

Some estimators expose configuration methods outside their constructors, such as `setValidationDataset()` on iterative learners. Use `setup()` to register a callback invoked on each newly instantiated base estimator before cross-validation and on the selected base estimator before full-dataset training.

The callback receives the base estimator instance. It is not serialized with the Genetic Search instance.

```php
use Rubix\ML\Classifiers\LogisticRegression;

$estimator = new GeneticSearch(LogisticRegression::class, $params);

$estimator->setup(function (LogisticRegression $classifier) use ($validation) : void {
    $classifier->setValidationDataset($validation);
});
```

## Additional Methods

Return the base learner instance.

```php
public base() : \Rubix\ML\Estimator
```

Return a Report of every unique hyper-parameter combination evaluated during the last search, with its validation score. The `Trial N` rows follow evaluation order, matching `scores()[N - 1]`. Each row contains the score under the metric's name and a nested `params` map of constructor parameters.

```php
public results() : \Rubix\ML\Report
```

Return validation scores in evaluation order.

```php
public scores() : ?array
```
