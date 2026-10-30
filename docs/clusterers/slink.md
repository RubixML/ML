<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/SLINK.php">[source]</a></span>

# SLINK

An *agglomerative* (bottom-up) hierarchical clustering algorithm that groups samples by the *single-link* criterion, named after the optimally efficient algorithm of *R. Sibson*. Unlike the iterative re-estimation schemes used by [K Means](k-means.md) or [K Medoids](k-medoids.md), SLINK begins with each sample as its own singleton cluster and then greedily merges the pair of clusters whose closest pair of members is nearest, repeating until the target number of clusters *k* remains. Merging halts early if the distance of the next closest pair of clusters exceeds `maxDistance`, which — defaulted to `INF` — permits the full *n - k* merges.

The distance kernel and the data types the estimator is compatible with are those of the tree. By default, SLINK uses a [Ball Tree](../graph/trees/ball-tree.md) with a Euclidean kernel, which accepts any *subadditive* distance kernel. A [KD Tree](../graph/trees/k-d-tree.md) can be substituted when a *monotonic* kernel is required. The algorithm is fully deterministic and requires no seeding, initialization, or epoch tuning.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Persistable](../persistable.md), [Verbose](../verbose.md)

**Data Type Compatibility:** Depends on distance kernel of the tree

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | k | | int | The number of target clusters. |
| 2 | maxDistance | INF | float | The maximum single-link distance any single merge may incur. Training stops early once the next closest pair of clusters exceeds this threshold. Defaults to `INF` (no limit). |
| 3 | tree | BallTree | Spatial | The spatial tree used to store the training samples and their labels and to search for nearest neighbors. |

## Example

```php
use Rubix\ML\Clusterers\SLINK;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = new SLINK(3, INF, new BallTree(kernel: new Manhattan()));
```

## Additional Methods

Return the spatial tree that stores the training samples and their cluster labels.

```php
public tree() : Spatial
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
