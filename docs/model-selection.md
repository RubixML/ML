# Model Selection
Model selection is the data-driven approach to choosing the hyper-parameters of a model. It aims to find the best combination that maximizes a particular cross validation [Metric](cross-validation/metrics/api.md) such as [Accuracy](cross-validation/metrics/accuracy.md), [F-Beta](cross-validation/metrics/f-beta.md), or the negative [Mean Squared Error](cross-validation/metrics/mean-squared-error.md). The [Grid Search](grid-search.md) meta-estimator automates model selection by training and testing a unique model for each combination of user-defined hyper-parameters. As an example, we could attempt to find the best setting for the hyper-parameter *k* in [K Nearest Neighbors](classifiers/k-nearest-neighbors.md) from a list of possible values `1`, `3`, `5`, and `10`. In addition, we could try each value of *k* with weighting turned `on` and `off`. We might also want to know if the underlying distance kernel makes a difference so we'll try both [Euclidean](https://docs.rubixml.com/en/latest/kernels/distance/euclidean.html) and [Manhattan](https://docs.rubixml.com/en/latest/kernels/distance/manhattan.html) distances. The order in which the parameters are given to Grid Search is the same order they appear in the constructor of the learner.

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

Once training is complete, Grid Search will automatically wrap and train the base learner with the best hyper-parameters on the full dataset. Then, it can either be used to perform inference like a normal estimator or you can dump the best parameter values for future reference.

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
In contrast to the *manual* search shown in the example above, when the values of the possible hyper-parameters are generated such that they are spaced out evenly, we call that *grid search*. You can use the static `grid()` method on the [Params](other/helpers/params.md) helper to generate an array of evenly-spaced values automatically. Instead of choosing the values of *k* manually, for this example we'll generate a grid of values between 1 and 10 with a grid spacing of 2.

```php
use Rubix\ML\Other\Helpers\Params;

$params = [
    Params::grid(1, 10, 2), [true, false], // ...
];
```

## Random Search
When the list of possible hyper-parameters is randomly chosen from a distribution, we call that *random search*. In the absence of a good manual strategy, random search has the advantage of being able to search the hyper-parameter space more effectively by testing combinations of parameters that might not have been considered otherwise. To generate a list of random values from a uniform distribution you can use either the `ints()` or `floats()` method on the [Params](other/helpers/params.md) helper. In the example below, we'll generate 5 unique random integers between 1 and 10 as possible values for *k*.

```php
use Rubix\ML\Other\Helpers\Params;

$params = [
    Params::ints(1, 10, 5), [true, false], // ...
];
```