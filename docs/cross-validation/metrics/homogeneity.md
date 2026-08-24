<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/Homogeneity.php">[source]</a></span>

# Homogeneity
A ground-truth clustering metric that measures how well each cluster is comprised of samples from a single class. A clustering is said to be *homogeneous* when all of its clusters contain only samples of a single class. Formally, it is one minus the conditional entropy of the classes given the cluster assignments normalized by the marginal entropy of the classes.

$$
{\displaystyle Homogeneity = 1-\frac{H(C \mid K)}{H(C)}}
$$

Even though each cluster contains a majority of samples from a single class, both clusters mix classes and the score is close to zero. A clustering is homogeneous only when every cluster contains samples from exactly one class. See [V Measure](v-measure.md) for the balanced combination of homogeneity and completeness, or [Cluster Purity](cluster-purity.md) for the purity-based counterpart of this metric.

!!! note
    Since this metric monotonically improves as the number of target clusters increases, it should not be used as a metric to guide hyper-parameter tuning.

!!! note
    When the ground-truth contains only one class, homogeneity is defined as 1 since there is no class entropy left to explain. An empty set of predictions scores 0.

**Estimator Compatibility:** Clusterer

**Score Range:** 0 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\Homogeneity;

$metric = new Homogeneity();

$score = $metric->score([0, 1, 1, 0, 1], ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

echo $score;
```

```
0.020570659450693
```
