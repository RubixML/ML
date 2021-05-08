<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/KMeans.php">[source]</a></span>

# K Means
A fast online centroid-based hard clustering algorithm capable of grouping linearly separable data points given some prior knowledge of the target number of clusters (defined by *k*). K Means is trained using adaptive Mini Batch Gradient Descent and minimizes the inertia cost function at each epoch. Inertia is defined as the average sum of distances between each sample and its nearest cluster centroid.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Persistable](../persistable.md), [Verbose](../verbose.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | | int | The number of target clusters. |
| 2 | batch size | 128 | int | The size of each mini batch in samples. |
| 3 | epochs | 1000 | int | The maximum number of training rounds to execute. |
| 4 | min change | 1e-4 | float | The minimum change in the inertia for training to continue. |
| 5 | window | 5 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 6 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |
| 7 | seeder | PlusPlus | Seeder | The seeder used to initialize the cluster centroids. |

## Example
```php
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\PlusPlus;

$estimator = new KMeans(3, 128, 300, 10.0, 10, new Euclidean(), new PlusPlus());
```

## Additional Methods
Return the *k* computed centroids of the training set:
```php
public centroids() : array[]
```

Return the number of training samples that each centroid is responsible for:
```php
public sizes() : int[]
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

Return the loss for each epoch from the last training session:
```php
public losses() : float[]|null
```

## References
[^1]: D. Sculley. (2010). Web-Scale K-Means Clustering.
