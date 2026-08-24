<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/Completeness.php">[source]</a></span>

# Completeness
A ground-truth clustering metric that measures how well all the samples of a class are grouped into a single cluster. A clustering is said to be *complete* when every sample of a class is contained in one cluster. Formally, it is one minus the conditional entropy of the cluster assignments given the classes normalized by the marginal entropy of the cluster assignments.

$$
{\displaystyle Completeness = 1-\frac{H(K \mid C)}{H(K)}}
$$

Assigning every sample its own cluster keeps the clusters perfectly pure (homogeneity 1) but scatters each class across multiple clusters, so completeness is heavily penalized. See [V Measure](v-measure.md) for the balanced combination of homogeneity and completeness, or [Class Purity](class-purity.md) for the purity-based counterpart of this metric.

!!! note
    Since this metric monotonically improves as the number of target clusters decreases, it should not be used as a metric to guide hyper-parameter tuning.

!!! note
    When all samples share a single cluster assignment, completeness is defined as 1 since there is no cluster entropy left to explain. An empty set of predictions scores 0.

**Estimator Compatibility:** Clusterer

**Score Range:** 0 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\Completeness;

$metric = new Completeness();

$score = $metric->score([0, 1, 2, 3, 4], ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

echo $score;
```

```
0.41816566007905
```
