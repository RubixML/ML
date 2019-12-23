<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/Homogeneity.php">[source]</a></span>

# Homogeneity
A ground-truth clustering metric that measures the ratio of samples in a cluster that are also members of the same class. A cluster is said to be *homogeneous* when the entire cluster is comprised of a single class of samples.

**Estimator Compatibility:** Clusterer

**Output Range:** 0 to 1

## Example
```php
use Rubix\ML\CrossValidation\Metrics\Homogeneity;

$metric = new Homogeneity();
```