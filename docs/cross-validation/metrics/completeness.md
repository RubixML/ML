<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/Completeness.php">[source]</a></span>

# Completeness
A ground-truth clustering metric that measures the ratio of samples in a class that are also members of the same cluster. A cluster is said to be *complete* when all the samples in a class are contained in a cluster.

$$
{\displaystyle Completeness = 1-\frac{H(K, C)}{H(K)}}
$$

!!! note
    Since this metric monotonically improves as the number of target clusters decreases, it should not be used as a metric to guide hyper-parameter tuning.

**Estimator Compatibility:** Clusterer

**Output Range:** 0 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\Completeness;

$metric = new Completeness();
```