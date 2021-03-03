<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/MeanShift.php">[source]</a></span>

# Mean Shift
A hierarchical clustering algorithm that uses peak (maxima) finding to locate the candidate centroids of a training set given a radius constraint. Near-duplicate centroids are merged together and the algorithm iterates on the remaining candidates in subsequent steps until the centroids stabilize.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | radius | | float | The bandwidth of the radial basis function. |
| 2 | ratio | 0.1 | float | The ratio of samples from the training set to use as initial centroids. |
| 3 | epochs | 100 | int | The maximum number of training rounds to execute. |
| 4 | minShift | 1e-4 | float | The minimum shift in the position of the centroids necessary to continue training. |
| 5 | tree | BallTree | Spatial | The spatial tree used to run range searches. |
| 6 | seeder | Random | Seeder | The seeder used to initialize the cluster centroids. |

## Example
```php
use Rubix\ML\Clusterers\MeanShift;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Clusterers\Seeders\KMC2;

$estimator = new MeanShift(2.5, 2000, 1e-6, 0.05, new BallTree(100), new KMC2());
```

## Additional Methods
Estimate the radius of a cluster that encompasses a certain percentage of the total training samples:
```php
public static estimateRadius(Dataset $dataset, float $percentile = 30.0, ?Distance $kernel = null) : float
```

!!! note
    Since radius estimation scales quadratically in the number of samples, for large datasets you can speed up the process by running it on a smaller subset of the training data.

Return the centroids computed from the training set:
```php
public centroids() : array[]
```

Returns the amount of centroid shift during each epoch of training:
```php
public steps() : float[]|null
```

## References
[^1]: M. A. Carreira-Perpinan et al. (2015). A Review of Mean-shift Algorithms for Clustering.
[^2]: D. Comaniciu et al. (2012). Mean Shift: A Robust Approach Toward Feature Space Analysis.