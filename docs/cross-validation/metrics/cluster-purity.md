<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/ClusterPurity.php">[source]</a></span>

# Cluster Purity
A ground-truth clustering metric that measures the mean ratio of samples in a cluster that are also members of the cluster's dominant class. A clustering is said to be *pure* when every cluster contains only samples of a single class.

$$
{\displaystyle Cluster\,Purity = {\frac {1}{k}}\sum _{i=1}^{k}{\frac {\max _{j}n_{ij}}{n_{i}}}}
$$

!!! note
    Since this metric monotonically improves as the number of target clusters increases, it should not be used as a metric to guide hyper-parameter tuning.

**Estimator Compatibility:** Clusterer

**Score Range:** 0 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\ClusterPurity;

$metric = new ClusterPurity();
```

Unlike [Homogeneity](homogeneity.md), this metric does not use conditional entropy and tends to give more lenient scores on mixed assignments. See [V Measure](v-measure.md) for the balanced entropy-based alternative.
