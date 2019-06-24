### Mean Shift
A hierarchical clustering algorithm that uses peak finding to locate the local maxima (*centroids*) of a training set given by a radius constraint.

> **Note**: Seeding Mean Shift with a [Seeder](#seeders) can speed up convergence using large datasets. The default is to initialize all training samples as seeds.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Clusterers/MeanShift.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | radius | | float | The bandwidth of the radial basis function. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between samples. |
| 3 | max leaf size | 30 | int | The max number of samples in a leaf node (*ball*). |
| 4 | epochs | 100 | int | The maximum number of training rounds to execute. |
| 5 | min change | 1e-4 | float | The minimum change in centroids necessary for the algorithm to continue training. |
| 6 | seeder | None | object | The seeder used to initialize the cluster centroids. |
| 7 | ratio | 0.2 | float | The ratio of samples from the training set to seed the algorithm with. |

**Additional Methods:**

Estimate the radius of a cluster that encompasses a certain percentage of the total training samples:
```php
public static estimateRadius(Dataset $dataset, float $percentile = 30., ?Distance $distance = null) : float
```

> **Note**: Since radius estimation scales quadratically in the number of samples, for large datasets you can speed up the process by running it on a sample subset of the training data.

Return the centroids computed from the training set:
```php
public centroids() : array
```

Returns the amount of centroid shift during each epoch of training:
```php
public steps() : array
```

**Example:**

```php
use Rubix\ML\Clusterers\MeanShift;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Clusterers\Seeders\KMC2;

$radius = MeanShift::estimateRadius($dataset, 30., new Diagonal()); // Automatically choose radius

$estimator = new MeanShift($radius, new Diagonal(), 30, 2000, 1e-6, new KMC2(), 0.1);
```

**References:**

>- M. A. Carreira-Perpinan et al. (2015). A Review of Mean-shift Algorithms for Clustering.
>- D. Comaniciu et al. (2012). Mean Shift: A Robust Approach Toward Feature Space Analysis.