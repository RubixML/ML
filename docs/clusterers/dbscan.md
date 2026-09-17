<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/DBSCAN.php">[source]</a></span>

# DBSCAN

*Density-Based Spatial Clustering of Applications with Noise* (DBSCAN) is a clustering algorithm able to find non-linearly separable and arbitrarily-shaped clusters given a radius and density constraint. In addition, DBSCAN can flag outliers (noise samples) and thus be used as a quasi-anomaly detector.

During training, the algorithm is run once on the training set and the non-noisy clustered samples are stored in a spatial tree for fast inference. Unseen samples are then assigned to the cluster that is most common among the samples within *radius* of them, or as noise if no samples are within *radius* of them. When *weighted* is set to `true`, the vote of each neighbor is weighted inversely to its distance. The `minDensity` constraint is only applied during training and does not affect the assignment of unseen samples.

!!! note
    DBSCAN assigns noise samples to the cluster number `-1`.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | radius | 1.0 | float | The maximum distance between two points to be considered neighbors. |
| 2 | minDensity | 5 | int | The minimum number of points within radius of each other to form a cluster. |
| 3 | weighted | false | bool | Should we consider the distances of our nearest neighbors when making predictions? |
| 4 | tree | BallTree | Spatial | The spatial tree used to run range searches. |

## Example

```php
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new DBSCAN(3.0, 10, true, new BallTree(20, new Diagonal()));

$estimator->train($dataset);

$predictions = $estimator->predict($unseen);

$probabilities = $estimator->proba($unseen);
```

## Additional Methods

Return the base spatial tree instance.

```php
public tree() : Spatial
```

## References

[^1]: M. Ester et al. (1996). A Density-Based Algorithm for Discovering Clusters.
