<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/Homogeneity.php">[source]</a></span>

# Homogeneity
A ground-truth clustering metric that measures the ratio of samples in a cluster that are also members of the same class. A cluster is said to be *homogeneous* when the entire cluster is comprised of a single class of samples.

> **Note:** Since homogeneity monotonically improves as the number of target clusters increases, it should not be used as a metric for hyper-parameter tuning.

**Estimator Compatibility:** Clusterer

**Output Range:** 0 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\Homogeneity;

$metric = new Homogeneity();
```