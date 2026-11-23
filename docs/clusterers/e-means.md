<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/EMeans.php">[source]</a></span>

# E Means

A mini batch centroid clustering algorithm in the spirit of E-Means, the analysis of the scaling laws of mini batch K-Means. Rather than being fixed by hand, the batch size and the initial exponential moving average (EMA) step are derived from the geometry of the dataset with fitted scaling laws, and the EMA step is then annealed along a cosine schedule over the course of the training run. Each mini batch updates the per-cluster sums and masses of the exponential moving average, clusters that fall below a fraction of the mean mass are re-seeded from the farthest members of the heaviest cluster, and the centroids are re-derived as the exact means of the members of the closing pass.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md), [Verbose](../verbose.md)

**Data Type Compatibility:** Continuous

**Requirements:** The Tensor C extension 4.1.0 or higher.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | k | | int | The number of target clusters. |
| 2 | epochs | 5.0 | float | The number of training passes over the dataset; a value below one restricts training to a fractional prefix. |
| 3 | batchSize | null | int | The size of each mini batch in samples, or `null` to derive it from the dataset with the batch laws. |
| 4 | alpha0 | null | float | The initial EMA step, or `null` to derive it from the dataset with the step law. |
| 5 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |
| 6 | seeder | Random | Seeder | The seeder used to initialize the cluster centroids. |

## Example

```php
use Rubix\ML\Clusterers\EMeans;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\Random;

$estimator = new EMeans(3, 5.0, null, null, new Euclidean(), new Random());
```

## Additional Methods

Return the *k* computed centroids of the training set.

```php
public centroids() : array[]
```

Return the number of training samples that each centroid is responsible for.

```php
public sizes() : int[]
```

Return the loss for each epoch from the last training session.

```php
public losses() : float[]|null
```

Return the progress table combining every epoch recorded during the last training session — the inertia at each epoch — into a single ordered sequence.

```php
public progress() : iterable
```

## References

[^1]: E-Means: Scaling Laws for Mini-Batch K-Means. https://arxiv.org/abs/2611.09501
