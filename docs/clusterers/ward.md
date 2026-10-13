<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/Ward.php">[source]</a></span>

# Ward

An *agglomerative* (bottom-up) hierarchical clustering algorithm named after *J. H. Ward, Jr.*. Unlike the iterative re-estimation schemes used by [K Means](k-means.md) or [K Medoids](k-medoids.md), Ward begins with each sample as its own singleton cluster and then greedily merges the pair of clusters whose union causes the smallest increase in the total *within-cluster sum of squares*, repeating until the target number of clusters *k* remains. The increase in within-cluster variance incurred by each merge is tracked with the **Lance-Williams** recurrence, allowing the full cost matrix to be updated in linear time. Because this objective only holds for the Euclidean distance, the distance kernel is restricted to `Euclidean`. The algorithm is fully deterministic and requires no seeding, initialization, or epoch tuning.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md), [Verbose](../verbose.md)

**Data Type Compatibility:** Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | k | | int | The number of target clusters. |
| 2 | kernel | Euclidean | Distance | The distance kernel, restricted to `Euclidean`, used to compute the distance between sample points. |

## Example

```php
use Rubix\ML\Clusterers\Ward;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new Ward(3, new Euclidean());
```

## Additional Methods

Return the *k* computed centroids of the training set.

```php
public centroids() : array[]|null
```

Return the number of training samples that each cluster is responsible for.

```php
public sizes() : int[]|null
```

Return the merge cost — the increase in the total within-cluster sum of squares — for each of the *n - k* merges from the last training session.

```php
public losses() : float[]|null
```

Return the progress table combining the merge cost recorded at each agglomeration round of the last training session into a single ordered sequence.

```php
public progress() : iterable
```

## References

[^1]: J.H. Ward. (1963). Hierarchical Grouping to Optimize an Objective Function.
