<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/Completeness.php">[source]</a></span>

# Completeness
A ground-truth clustering metric that measures the ratio of samples in a class that are also members of the same cluster. A cluster is said to be *complete* when all the samples in a class are contained in a cluster.

**Estimator Compatibility:** Clusterer

**Output Range:** 0 to 1

## Example
```php
use Rubix\ML\CrossValidation\Metrics\Completeness;

$metric = new Completeness();
```