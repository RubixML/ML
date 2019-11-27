# Model Selection
Model selection is the data-driven approach to choosing the hyper-parameters of a model. It aims to find the best model that maximizes a particular cross validation [Metric](cross-validation/metrics/api.md) such as [Accuracy](cross-validation/metrics/accuracy.md), [F Beta](cross-validation/metrics/f-beta.md), or the negative [Mean Squared Error](cross-validation/metrics/mean-squared-error.md). The Rubix ML library provides the [Grid Search](grid-search.md) meta-estimator that automates model selection by training and testing a unique model for each combination of user-defined hyper-parameters. As an example, let's say we wanted to find the best setting for *k* in [K Nearest Neighbors](classifiers/k-nearest-neighbors.md) from a list of possible values `1`, `3`, `5`, and `10`. In addition, let's try each value of *k* with weighting turned `on` and `off`. We might also want to know if the underlying distance kernel makes a difference so we'll try both Euclidean and Manhattan distances as well. The order in which the possible parameters are given to Grid Search is the same order that they appear in the constructor of the estimator.

```php
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;

// Import dataset with ground-truth labels

$params = [
    [1, 3, 5, 10], [true, false], [new Euclidean(), new Manhattan()]
];

$estimator = new GridSearch(KNearestNeighbors::class, $params);

$estimator->train($dataset);
```

Once training is complete, Grid Search will automatically wrap and train the base learner with the best hyper-parameters on the full dataset. Then, it can either be used to perform inference like a normal estimator or you can dump the best parameter values for reference.

```php
$predictions = $estimator->predict($dataset);
```

```php
var_dump($estimator->best());
```

```sh
array(3) {
  [0]=> int(3)
  [1]=> bool(true)
  [2]=> object(Rubix\ML\Kernels\Distance\Manhattan) {}
}
```

## Grid Search
When the list of possible hyper-parameters are spaced out evenly, we call that *grid search*. Instead of doing the math to choose these parameters in your head, you can use the [Params](other/helpers/params.md) helper to generate an evenly-spaced list of possible parameters for you. Here we'll choose the same params as the example above except that instead of choosing the values of *k* we'll generate a grid of values between 1 and 10 with a grid spacing of 2.

```php
use Rubix\ML\Other\Helpers\Params;

$params = [
    Params::grid(1, 20, 2), [true, false], // ...
];
```

## Random Search

On the map ...