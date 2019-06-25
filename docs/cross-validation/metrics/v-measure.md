<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/VMeasure.php">Source</a></span>

# V Measure
V Measure is the harmonic balance between [homogeneity](homogeneity.md) and [completeness](completeness.md) and is used as a measure to determine the quality of a clustering.

**Estimator Compatibility:** Clusterer

**Output Range:** 0 to 1

### Example
```php
use Rubix\ML\CrossValidation\Metrics\VMeasure;

$metric = new VMeasure();
```

### References
>- A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based external cluster evaluation measure.