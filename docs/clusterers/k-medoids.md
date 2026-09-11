<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/KMedoids.php">[source]</a></span>

# K Medoids

A robust medoid-based hard clustering algorithm capable of grouping linearly separable data points given some prior knowledge of the target number of clusters (defined by *k*). Unlike centroid-based algorithms such as [K Means](k-means.md), K Medoids anchors each cluster with an *actual* sample of the training set (called a *medoid*) rather than with a mean vector, making the resultant clustering less sensitive to outliers and noise.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md), [Verbose](../verbose.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | k | | int | The number of target clusters. |
| 2 | sample size | 100 | int | The number of samples drawn from the training set to propose a candidate set of medoids at each epoch. |
| 3 | epochs | 100 | int | The number of CLARA candidates to propose. The best candidate (lowest inertia) is kept. |
| 4 | min change | 1e-4 | float | The minimum improvement in the inertia required for a PAM SWAP exchange to be accepted. |
| 5 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |
| 6 | seeder | KMC2 | Seeder | The seeder used to initialize the cluster medoids. |

## Example

```php
use Rubix\ML\Clusterers\KMedoids;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\KMC2;

$estimator = new KMedoids(3, 100, 300, 1e-4, new Euclidean(), new KMC2());
```

## Additional Methods

Return the *k* computed medoids of the training set. Note that, unlike *centroids* computed by K Means, each returned medoid is an *exact copy* of an actual sample from the training data.

```php
public medoids() : array[]
```

Return the number of training samples that each medoid is responsible for.

```php
public sizes() : int[]
```

Return an iterable progress table with the steps from the last training session.

```php
public steps() : iterable
```

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->steps());
```

Return the full-dataset inertia of the best candidate medoid proposed at each CLARA epoch (i.e. the loss for each of the *epochs* candidates considered).

```php
public losses() : float[]|null
```

## References

[^1]: A. K. Jain et al. (1999). Data Clustering: A Review.
