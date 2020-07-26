# AutoML
Automated Machine Learning (AutoML) is the application of automated tools in designing machine learning models. Some examples of [automated machine learning](https://en.wikipedia.org/wiki/Automated_machine_learning) include feature selection, model selection, and hyper-parameter optimization. The benefits of AutoML include quicker prototyping and the discovery of simpler and more accurate solutions than could otherwise be discovered through human intuition.

## Hyper-parameter Optimization
Hyper-parameter optimization aims to find the best combination of hyper-parameters that maximize a particular cross-validation [Metric](cross-validation/metrics/api.md). The [Grid Search](grid-search.md) meta-estimator learns the best hyper-parameters by training and testing a unique model for each combination of hyper-parameters. Since Grid Search implements the [Parallel](parallel.md) interface, each model can be trained on its own CPU core in parallel.

As an example, we could attempt to find the best setting for the hyper-parameter *k* in [K Nearest Neighbors](classifiers/k-nearest-neighbors.md) from a list of possible values `1`, `3`, `5`, and `10`. In addition, we could try each value of *k* with distance weighting turned on or off. We might also want to know if the data is sensitive to the underlying distance kernel so we'll try the standard [Euclidean](https://docs.rubixml.com/en/latest/kernels/distance/euclidean.html) as well as the [Manhattan](https://docs.rubixml.com/en/latest/kernels/distance/manhattan.html) distances. The order in which the sets of possible parameters are given to Grid Search is the same order they are given in the constructor of the learner.

```php
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;

$params = [
    [1, 3, 5, 10], [true, false], [new Euclidean(), new Manhattan()]
];

$estimator = new GridSearch(KNearestNeighbors::class, $params);

$estimator->train($dataset);
```

Once training is complete, Grid Search automatically trains the base learner with the best hyper-parameters on the full dataset and can perform inference like a normal estimator. In addition, you can dump the results of the search for future reference using the `results()` method. In the example below, we'll return just the parameters that received the highest validation score using the `best()` method.

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

To return the base estimator call the `base()` method.

```php
$base = $estimator->base();
```

### Grid Search
When the possible values of the hyper-parameters are selected such that they are spaced out evenly, we call that *grid search*. You can use the static `grid()` method on the [Params](other/helpers/params.md) helper to generate an array of evenly-spaced values automatically. Instead of choosing the values of *k* manually, for this example we'll generate a grid of 4 values equally spaced between 1 and 10.

```php
use Rubix\ML\Other\Helpers\Params;

$params = [
    Params::grid(1, 10, 4), [true, false], // ...
];
```

### Random Search
When the list of possible hyper-parameters is randomly chosen from a distribution, we call that *random search*. In the absence of a good manual strategy, random search has the advantage of being able to search the hyper-parameter space more effectively by testing combinations of parameters that might not have been considered otherwise. To generate a list of random values from a uniform distribution you can use either the `ints()` or `floats()` method on the [Params](other/helpers/params.md) helper. In the example below, we'll generate 4 unique random integers between 1 and 10 as possible values for *k*.

```php
use Rubix\ML\Other\Helpers\Params;

$params = [
    Params::ints(1, 10, 4), [true, false], // ...
];
```