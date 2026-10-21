<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/SLINK.php">[source]</a></span>

# SLINK

An *agglomerative* (bottom-up) hierarchical clustering algorithm that groups samples by the *single-link* (nearest neighbor) criterion, named after the optimally efficient algorithm of *R. Sibson*. Unlike the iterative re-estimation schemes used by [K Means](k-means.md) or [K Medoids](k-medoids.md), SLINK begins with each sample as its own singleton cluster and then greedily merges the pair of clusters whose closest pair of members is nearest, repeating until the target number of clusters *k* remains. Merging halts early if the distance of the next closest pair of clusters exceeds `maxDistance`, which — defaulted to `INF` — permits the full *n - k* merges.

Rather than storing the pairwise distance matrix, SLINK maintains a compact *pointer representation* of the dendrogram in two length-n arrays, refreshing it in amortized linear time as each sample is added. This yields a total running time of O(n²) with only O(n) auxiliary space — optimal in both bounds — making the single-link method feasible for datasets well into the thousands of samples.

Because single-link clusters are connected by chains of nearby samples, they tend to be elongated and are therefore anchored by a *medoid* — the actual training sample with the lowest total distance to the rest of its cluster — rather than a mean vector. Unlike [Ward](ward.md), which is restricted to the squared Euclidean distance, SLINK is valid for any symmetric distance kernel. The algorithm is fully deterministic and requires no seeding, initialization, or epoch tuning.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md), [Verbose](../verbose.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | k | | int | The number of target clusters. |
| 2 | maxDistance | INF | float | The maximum single-link distance any single merge may incur. Training stops early once the next closest pair of clusters exceeds this threshold. Defaults to `INF` (no limit). |
| 3 | kernel | Euclidean | Distance | The symmetric distance kernel to use when computing the distance between samples. |

## Example

```php
use Rubix\ML\Clusterers\SLINK;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = new SLINK(3, INF, new Manhattan());
```

## Additional Methods

Return the *k* computed medoids of the training set i.e. the actual samples that anchor each cluster.

```php
public medoids() : array[]|null
```

Return the number of training samples that each cluster is responsible for.

```php
public sizes() : int[]|null
```

Return the single-link merge height — the distance between the closest pair of members of the two clusters — for each of the merges from the last training session.

```php
public losses() : float[]|null
```

Return the progress table combining the merge height recorded at each agglomeration round of the last training session into a single ordered sequence.

```php
public progress() : iterable
```

## References

[^1]: R. Sibson. (1973). SLINK: An Optimally Efficient Algorithm for the Single-Link Cluster Method. The Computer Journal, 16(1), 30-34.
