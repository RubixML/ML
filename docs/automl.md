# AutoML
Automated Machine Learning (AutoML) is the application of automated tools when designing machine learning models. The goal of AutoML is to simplify the machine learning lifecyle for non-experts and to facilitate rapid prototyping. In addition, AutoML can aid in the discovery of simpler and more accurate solutions than could otherwise be discovered via human intuition.

## Hyper-parameter Optimization
Hyper-parameter optimization is a way of automating the process of hyper-parameter tuning. It is a data-driven approach to choosing the best hyper-parameters of a model. 

### Hyper-parameter Search
[Grid Search](grid-search.md) is a meta-estimator that aims to find the best combination of hyper-parameters that maximizes a particular cross-validation [Metric](cross-validation/metrics/api.md). It is a black box optimizer that works by training and testing a unique model for each combination of possible user-defined hyper-parameters and then picking the ones that returned the highest validation score. Since Grid Search implements the [Parallel](parallel.md) interface, each model can be trained on its own CPU core in parallel greatly reducing the search time for each model.

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

#### Grid Search
When the possible values of the hyper-parameters are selected such that they are spaced out evenly, we call that *grid search*. You can use the static `grid()` method on the [Params](other/helpers/params.md) helper to generate an array of evenly-spaced values automatically.

```php
use Rubix\ML\Other\Helpers\Params;

$params = [
    Params::grid(1, 10, 4), [true, false], // ...
];
```

#### Random Search
When the list of possible hyper-parameters is randomly chosen from a distribution, we call that *random search*. In the absence of a good manual strategy, random search has the advantage of being able to search the hyper-parameter space more effectively by testing combinations of parameters that might not have been considered otherwise. To generate a list of random values from a uniform distribution you can use either the `ints()` or `floats()` method on the [Params](other/helpers/params.md) helper.

```php
use Rubix\ML\Other\Helpers\Params;

$params = [
    Params::ints(1, 10, 4), [true, false], // ...
];
```

## Feature Selection
The task of choosing the features to train a learner on that will be most informative or correlated with the outcome is known as feature selection. Having fewer but better features can help to reduce overfitting, reduce training times, and by making models easier to interpret. To help automate the feature selection process, the library provides a number of feature selectors that implement the [Transformer](transformers/api.md) interface and thus can be applied directly to dataset objects.

### Recursive Feature Elimination
[Recursive Feature Elimination](transformers/recursive-feature-eliminator.md) (RFE) is an iterative feature selection technique that scores a set of features using a learner that implements the [Ranks Features](ranks-features.md) interface, drops the lowest scoring features, and then recursively repeats the process on the smaller subsets of features until it can no longer make any progress. In the example below we'll use a [Random Forest](classifiers/random-forest.md) as the base feature ranker to select the top 7 features from a dataset and then dump their importance scores.

```php
use Rubix\ML\Transformers\RecursiveFeatureEliminator;
use Rubix\ML\Classifiers\RandomForest;

$transformer = new RecursiveFeatureEliminator(7, 10, 0.1, new RandomForest());

$dataset->apply($transformer);

$importances = $transformer->importances();

arsort($importances);

print_r($importances);
```

```sh
Array
(
    [19] => 0.1903367387308
    [28] => 0.18110555366077
    [11] => 0.15159278156525
    [38] => 0.14015492789444
    [35] => 0.13451993746636
    [39] => 0.11995324726519
    [14] => 0.082336813417187
)
```
