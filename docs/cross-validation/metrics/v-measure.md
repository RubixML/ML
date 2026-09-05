<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/VMeasure.php">[source]</a></span>

# V Measure
V Measure is an entropy-based clustering metric that balances [Homogeneity](homogeneity.md) and [Completeness](completeness.md). It has the additional property of being symmetric in that the predictions and ground-truth can be swapped without changing the score when beta is left at its default value of 1.

$$
{\displaystyle V_{\beta} = \frac{(1+\beta)hc}{\beta h + c}}
$$

!!! note
    A beta greater than 1 gives more weight to homogeneity while a beta less than 1 favors completeness. Since V Measure is a harmonic mean, the score is 0 whenever either homogeneity or completeness is 0.

!!! note
    Unlike Homogeneity and Completeness on their own, V Measure can be used to guide hyper-parameter tuning and is the default scoring metric chosen by [Grid Search](../../grid-search.md) for clusterers.

**Estimator Compatibility:** Clusterer

**Score Range:** 0 to 1

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | beta | 1.0 | float | The ratio of weight given to homogeneity over completeness. |

## Example

```php
use Rubix\ML\CrossValidation\Metrics\VMeasure;

$metric = new VMeasure(1.0);

$score = $metric->score([0, 1, 1, 0, 1], ['lamb', 'lamb', 'wolf', 'wolf', 'wolf']);

echo $score;
```

```
0.020570659450693
```

## References

[^1]: A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based external cluster evaluation measure.
