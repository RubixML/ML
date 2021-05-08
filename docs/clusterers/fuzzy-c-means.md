<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/FuzzyCMeans.php">[source]</a></span>

# Fuzzy C Means
A distance-based soft-clustering algorithm that allows samples to belong to multiple clusters if they fall within a *fuzzy* region controlled by the fuzz hyper-parameter. Like [K Means](k-means.md), Fuzzy C Means minimizes the inertia cost function, however, unlike K Means, FCM uses a batch solver that requires the entire training set to compute the update to the cluster centroids at each step.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | c | | int | The number of target clusters. |
| 2 | fuzz | 2.0 | float | Determines the bandwidth of the fuzzy area. |
| 3 | epochs | 300 | int | The maximum number of training rounds to execute. |
| 4 | minChange | 1e-4 | float | The minimum change in the inertia for the algorithm to continue training. |
| 5 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |
| 6 | seeder | PlusPlus | Seeder | The seeder used to initialize the cluster centroids. |

## Example
```php
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\Random;

$estimator = new FuzzyCMeans(5, 1.2, 400, 1., new Euclidean(), new Random());
```

## Additional Methods
Return the *c* computed centroids of the training set:
```php
public centroids() : array[]
```

Return an iterable progress table with the steps from the last training session:
```php
public steps() : iterable
```

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->steps());
```

Returns the inertia at each epoch from the last round of training:
```php
public losses() : float[]|null
```

## References
[^1]: J. C. Bezdek et al. (1984). FCM: The Fuzzy C-Means Clustering Algorithm.
