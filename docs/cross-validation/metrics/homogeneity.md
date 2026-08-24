<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/Homogeneity.php">[source]</a></span>

# Homogeneity
A ground-truth clustering metric that measures how well each cluster is comprised of samples from a single class. A clustering is said to be *homogeneous* when all of its clusters contain only samples of a single class. Formally, it is one minus the conditional entropy of the classes given the cluster assignments normalized by the marginal entropy of the classes.

$$
{\displaystyle Homogeneity = 1-\frac{H(C \mid K)}{H(C)}}
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