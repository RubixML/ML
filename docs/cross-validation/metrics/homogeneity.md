<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/Homogeneity.php">[source]</a></span>

# Homogeneity
A ground-truth clustering metric that measures the ratio of samples in a cluster that are also members of the same class. A cluster is said to be *homogeneous* when the entire cluster is comprised of a single class of samples.

$$
{\displaystyle Homogeneity = 1-\frac{H(C, K)}{H(C)}}
$$

!!! note
    Since this metric monotonically improves as the number of target clusters increases, it should not be used as a metric to guide hyper-parameter tuning.

**Estimator Compatibility:** Clusterer

**Score Range:** 0 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\Homogeneity;

$metric = new Homogeneity();
```