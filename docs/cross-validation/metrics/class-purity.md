<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/ClassPurity.php">[source]</a></span>

# Class Purity
A ground-truth clustering metric that measures the mean ratio of samples in a class that are also members of the class' dominant cluster. A clustering is said to be *complete* when all the samples in a class are contained in a single cluster.

$$
{\displaystyle Class\,Purity = {\frac {1}{m}}\sum _{j=1}^{m}{\frac {\max _{i}n_{ij}}{n_{j}}}}
$$

!!! note
    Since this metric monotonically improves as the number of target clusters decreases, it should not be used as a metric to guide hyper-parameter tuning.

**Estimator Compatibility:** Clusterer

**Score Range:** 0 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\ClassPurity;

$metric = new ClassPurity();
```
