<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/KNNRegressor.php">[source]</a></span>

# KNN Regressor

K Nearest Neighbors (KNN) is a brute-force distance-based learner that locates the k nearest training samples from the training set and averages their labels to make a prediction. K Nearest Neighbors (KNN) is considered a *lazy* learner because it performs most of its computation at inference time.

!!! note
    For a faster spatial tree-accelerated version of KNN, see [KD Neighbors Regressor](kd-neighbors-regressor.md).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Parallel](../parallel.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | k | 5 | int | The number of nearest neighbors to consider when making a prediction. |
| 2 | weighted | false | bool | Should we consider the distances of our nearest neighbors when making predictions? |
| 3 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |

## Example

```php
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Kernels\Distance\SafeEuclidean;

$estimator = new KNNRegressor(5, false, new SafeEuclidean());
```

## Parallel

This estimator implements the [Parallel](../parallel.md) interface and can utilize a parallel processing backend such as [Swoole](../backends/swoole.md) to speed up training:

```php
use Rubix\ML\Backends\Swoole;

$estimator->setBackend(new Swoole());
```

## Additional Methods

This estimator does not have any additional methods.
