<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/VMeasure.php">[source]</a></span>

# V Measure
V Measure is an entropy-based clustering metric that balances [Homogeneity](homogeneity.md) and [Completeness](completeness.md). It has the additional property of being symmetric in that the predictions and ground-truth can be swapped without changing the score.

**Estimator Compatibility:** Clusterer

**Output Range:** 0 to 1

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | beta | 1.0 | float | The ratio of weight given to homogeneity over completeness. |

## Example
```php
use Rubix\ML\CrossValidation\Metrics\VMeasure;

$metric = new VMeasure(1.0);
```

### References
>- A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based external cluster evaluation measure.