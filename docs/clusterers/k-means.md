<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Clusterers/KMeans.php">Source</a></span></p>

# K Means
A fast online centroid-based hard clustering algorithm capable of clustering linearly separable data points given some prior knowledge of the target number of clusters (defined by *k*). K Means with inertia is trained using adaptive mini batch gradient descent and minimizes the inertial cost function. Inertia is defined as the sum of the distances between each sample and its nearest cluster centroid.

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Online](#online), [Probabilistic](#probabilistic), [Persistable](#persistable), [Verbose](#verbose)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | | int | The number of target clusters. |
| 2 | batch size | 100 | int | The size of each mini batch in samples. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 4 | epochs | 300 | int | The maximum number of training rounds to execute. |
| 5 | min change | 10. | float | The minimum change in the inertia for training to continue. |
| 6 | seeder | PlusPlus | object | The seeder used to initialize the cluster centroids. |

### Additional Methods
Return the *k* computed centroids of the training set:
```php
public centroids() : array
```

Return the number of training samples that each centroid is responsible for:
```php
public sizes() : array
```

Return the value of the inertial function at each epoch from the last round of training:
```php
public steps() : array
```

### Example
```php
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Kernels\Distance\Euclidean;

$estimator = new KMeans(3, 100, new Euclidean(), 300, 10., new PlusPlus());
```

### References
>- D. Sculley. (2010). Web-Scale K-Means Clustering.