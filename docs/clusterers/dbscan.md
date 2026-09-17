<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/DBSCAN.php">[source]</a></span>

# DBSCAN

*Density-Based Spatial Clustering of Applications with Noise* (DBSCAN) is a clustering algorithm able to find non-linearly separable and arbitrarily-shaped clusters given a radius and density constraint. In addition, DBSCAN can flag outliers (noise samples) and thus be used as a quasi-anomaly detector.

!!! note
    Noise samples are assigned the cluster number -1.

During training, the algorithm is run once on the training set and the clustered samples are stored in a spatial tree for fast inference. Unseen samples are then assigned to the cluster that is most common among the samples within *radius* of them, or as noise if no samples are within *radius* of them. The `minDensity` constraint is only applied during training and does not affect the assignment of unseen samples.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | radius | 1.0 | float | The maximum distance between two points to be considered neighbors. |
| 2 | minDensity | 5 | int | The minimum number of points within radius of each other to form a cluster. |
| 3 | tree | BallTree | Spatial | The spatial tree used to run range searches. |

## Example

```php
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new DBSCAN(4.0, 5, new BallTree(20, new Diagonal()));

$estimator->train($dataset);

$predictions = $estimator->predict($unseen);
```

## Additional Methods

Return the base spatial tree instance.

```php
public tree() : Spatial
```

## References

[^1]: M. Ester et al. (1996). A Density-Based Algorithm for Discovering Clusters.