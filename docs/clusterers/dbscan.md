<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Clusterers/DBSCAN.php">Source</a></span></p>

# DBSCAN
*Density-Based Spatial Clustering of Applications with Noise* is a clustering algorithm able to find non-linearly separable and arbitrarily-shaped clusters. In addition, DBSCAN also has the ability to mark outliers as *noise* and thus can be used as a *quasi* [Anomaly Detector](#anomaly-detectors).

> **Note**: Noise samples are assigned the cluster number *-1*.

**Interfaces:** [Estimator](#estimators)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | radius | 0.5 | float | The maximum distance between two points to be considered neighbors. |
| 2 | min density | 5 | int | The minimum number of points within radius of each other to form a cluster. |
| 3 | kernel | Euclidean | object | The distance kernel used to compute the distance between sample points. |
| 4 | max leaf size | 30 | int | The max number of samples in a leaf node (*ball*). |

> **Note**: The smaller the radius, the tighter the clusters will be.

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new DBSCAN(4.0, 5, new Diagonal(), 20);
```

### References
>- M. Ester et al. (1996). A Densty-Based Algorithm for Discovering Clusters.