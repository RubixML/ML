<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Clusterers/FuzzyCMeans.php">Source</a></span>

# Fuzzy C Means
Distance-based soft clusterer that allows samples to belong to multiple clusters if they fall within a *fuzzy* region controlled by the *fuzz* parameter.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | c | | int | The number of target clusters. |
| 2 | fuzz | 2.0 | float | Determines the bandwidth of the fuzzy area. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 4 | epochs | 300 | int | The maximum number of training rounds to execute. |
| 5 | min change | 10. | float | The minimum change in the inertia for the algorithm to continue training. |
| 6 | seeder | PlusPlus | object | The seeder used to initialize the cluster centroids. |

### Additional Methods
Return the *c* computed centroids of the training set:
```php
public centroids() : array
```

Returns the inertia at each epoch from the last round of training:
```php
public steps() : array
```

### Example
```php
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Clusterers\Seeders\Random;

$estimator = new FuzzyCMeans(5, 1.2, new Euclidean(), 400, 1., new Random());
```

### References
>- J. C. Bezdek et al. (1984). FCM: The Fuzzy C-Means Clustering Algorithm.