<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/VMeasure.php">[source]</a></span>

# V Measure
V Measure is an entropy-based clustering metric that balances [homogeneity](homogeneity.md) and [completeness](completeness.md).

**Estimator Compatibility:** Clusterer

**Output Range:** 0 to 1

### Example
```php
use Rubix\ML\CrossValidation\Metrics\VMeasure;

$metric = new VMeasure();
```

### References
>- A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based external cluster evaluation measure.