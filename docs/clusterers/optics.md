# OPTICS

*Ordering Points To Identify the Clustering Structure* (OPTICS) is a density-based clustering algorithm closely related to [DBSCAN](dbscan.md). Unlike DBSCAN, which relies on a single density threshold, OPTICS builds an ordering of the training set together with a *reachability* distance for each sample. Clusters are then extracted from the resulting reachability plot using the *Xi* (steepness) method, which allows clusters of varying density to be discovered in a single pass. In addition, OPTICS can flag outliers (noise samples) and thus be used as a quasi-anomaly detector.

During training, the algorithm is run once on the training set and the non-noisy clustered samples are stored in a spatial tree for fast inference. Unseen samples are then assigned to the cluster that is most common among the samples within *radius* of them, or as noise if no samples are within *radius* of them. When *weighted* is set to `true`, the vote of each neighbor is weighted inversely to its distance. The `minDensity` constraint is only applied during training and does not affect the assignment of unseen samples.

!!! note
    OPTICS assigns noise samples to the cluster number `-1`.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | radius | 1.0 | float | The maximum distance between two points to be considered neighbors. Bounds the reachability distances considered during training. |
| 2 | minDensity | 5 | int | The minimum number of points within radius of each other to form a dense region. |
| 3 | steepness | 0.05 | float | The minimum steepness on the reachability plot that constitutes a cluster boundary. |
| 4 | minClusterSize | null | int | The minimum number of samples contained within an extracted cluster. Defaults to *minDensity* when `null`. |
| 5 | weighted | false | bool | Should we consider the distances of our nearest neighbors when making predictions? |
| 6 | tree | BallTree | Spatial | The spatial tree used to run range searches. |

## Example

```php
use Rubix\ML\Clusterers\OPTICS;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new OPTICS(3.0, 10, 0.05, null, true, new BallTree(20, new Diagonal()));

$estimator->train($dataset);

$predictions = $estimator->predict($unseen);

$probabilities = $estimator->proba($unseen);
```

## Additional Methods

Return the base spatial tree instance.

```php
public tree() : Spatial
```

Return the cluster ordered list of sample indices from the last training session.

```php
public ordering() : array
```

Return the reachability distance of each sample, indexed by sample number, from the last training session.

```php
public reachability() : array
```

Return the core distance of each sample, indexed by sample number, from the last training session.

```php
public coreDistances() : array
```

## References

[^1]: M. Ankerst et al. (1999). OPTICS: Ordering Points To Identify the Clustering Structure.
[^2]: E. Schubert et al. (2018). Improving the Cluster Structure Extracted from OPTICS Plots.
