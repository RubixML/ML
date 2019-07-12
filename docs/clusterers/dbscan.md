<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Clusterers/DBSCAN.php">Source</a></span>

# DBSCAN
*Density-Based Spatial Clustering of Applications with Noise* is a clustering algorithm able to find non-linearly separable and arbitrarily-shaped clusters given a radius and density constraint. In addition, DBSCAN also has the ability to mark outliers as *noise* and thus can be used as a *quasi* anomaly detector.

> **Note:** The smaller the radius, the *tighter* the clusters will be.

> **Note:** Noise samples are assigned the cluster number -1.

**Interfaces:** [Estimator](../estimator.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | radius | 0.5 | float | The maximum distance between two points to be considered neighbors. |
| 2 | min density | 5 | int | The minimum number of points within radius of each other to form a cluster. |
| 3 | tree | BallTree | object | The spatial tree used for range queries. |

### Additional Methods
This estimator does not have any additional methods.

### Example
```php
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Diagonal;

$estimator = new DBSCAN(4.0, 5, new BallTree(20, new Diagonal()));
```

### References
>- M. Ester et al. (1996). A Densty-Based Algorithm for Discovering Clusters.