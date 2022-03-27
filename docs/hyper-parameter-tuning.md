# Hyper-parameter Tuning
Hyper-parameter tuning is an experimental process that incorporates [cross-validation](cross-validation.md) to guide hyper-parameter selection. When choosing an estimator for your project it often helps to fine-tune its hyper-parameters in order to get the best accuracy and performance from the model.

## Manual Tuning
When actively tuning a model, we will train an estimator with one set of hyper-parameters, obtain a validation score, and then use that as a baseline to make future adjustments. The goal at each iteration is to determine whether the adjustments improve accuracy or cause it to decrease. We can consider a model to be *fully* tuned when adjustments to the hyper-parameters can no longer make improvements to the validation score. With practice, we'll develop an intuition for which parameters need adjusting. Refer to the API documentation for each learner for a description of each hyper-parameter. In the example below, we'll tune the *radius* parameter of [Radius Neighbors Regressor](regressors/radius-neighbors-regressor.md) by iterating over the following block of code with a different setting each time. At first, we can start by choosing radius from a set of values and then honing in on the best value once we have obtained the settings with the highest [SMAPE](cross-validation/metrics/smape.md) score.

```php
use Rubix\ML\Regressors\RadiusNeighborsRegressor;
use Rubix\ML\CrossValidation\Metrics\SMAPE;

[$training, $testing] = $dataset->randomize()->split(0.8);

$estimator = new RadiusNeighborsRegressor(0.5); // 0.1, 0.5, 1.0, 2.0, 5.0

$estimator->train($training);

$predictions = $estimator->predict($testing);

$metric = new SMAPE();

$score = $metric->score($predictions, $testing->labels());

echo $score;
```

```
-4.75
```

### Deterministic Training
When the algorithm that trains a Learner is *stochastic* or randomized, it may be desirable for the sake of hyper-parameter tuning to isolate the effect of randomness on training. Fortunately, PHP makes it easy to seed the pseudo-random number generator (PRNG) with a known constant so your training sessions are repeatable. To seed the random number generator call the `srand()` function at the start of your training script passing any integer constant. After that point the PRNG will generate the same series of random numbers each time the training script is run.

```php
srand(42)
```

## Hyper-parameter Optimization
In distinction to manual tuning, Hyper-parameter optimization is an AutoML technique that employs search and meta-learning strategies to explore various algorithm configurations. In Rubix ML, hyper-parameter optimizers are implemented as meta-estimators that wrap a base learner whose hyper-parameters we wish to optimize.

### Grid Search
[Grid Search](grid-search.md) is a meta-estimator that aims to find the combination of hyper-parameters that maximizes a particular cross-validation [Metric](cross-validation/metrics/api.md). It works by training and testing a unique model for each combination of possible hyper-parameters and then picking the combination that returns the highest validation score. Since Grid Search implements the [Parallel](parallel.md) interface, we can greatly reduce the search time by training many models in parallel.

As an example, we could attempt to find the best setting for the hyper-parameter *k* in [K Nearest Neighbors](classifiers/k-nearest-neighbors.md) from a list of possible values `1`, `3`, `5`, and `10`. In addition, we could try each value of *k* with distance weighting turned on or off. We might also want to know if the data is sensitive to the underlying distance kernel so we'll try the standard [Euclidean](kernels/distance/euclidean.md) as well as the [Manhattan](kernels/distance/manhattan.md) distances. The order in which the sets of possible parameters are given to Grid Search is the same order they are given in the constructor of the learner.

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

Once training is complete, Grid Search automatically trains the base learner with the best hyper-parameters on the full dataset and can perform inference like a normal estimator.

```php
$predictions = $estimator->predict($dataset);
```

We can also dump the selected hyper-parameters by calling the `params()` method on the base learner. To return the base learner trained by Grid Search, call the `base()` method like in the example below.

```php
print_r($estimator->base()->params());
```

```php
Array
(
    [k] => 3
    [weighted] => true
    [kernel] => Rubix\ML\Kernels\Distance\Euclidean Object ()
)
```
### Grid Search vs. Random Search
When the possible values of the continuous hyper-parameters are selected such that they are evenly spaced out in a grid, we call that *grid search*. You can use the static `grid()` method on the [Params](helpers/params.md) helper to generate an array of evenly-spaced values automatically.

```php
use Rubix\ML\Helpers\Params;

$params = [
    Params::grid(1, 10, 4), [true, false], // ...
];
```

When the list of possible continuous-valued hyper-parameters is randomly chosen from a distribution, we call that *random search*. In the absence of a good manual strategy, random search has the advantage of being able to search the hyper-parameter space more effectively by testing combinations of parameters that might not have been considered otherwise. To generate a list of random values from a uniform distribution you can use either the `ints()` or `floats()` method on the [Params](helpers/params.md) helper.

```php
use Rubix\ML\Helpers\Params;

$params = [
    Params::ints(1, 10, 4), [true, false], // ...
];
```
