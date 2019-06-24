### V Measure
V Measure is the harmonic balance between [homogeneity](#homogeneity) and [completeness](#completeness) and is used as a measure to determine the quality of a clustering.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/VMeasure.php)

**Compatibility:** Clustering

**Range:** 0 to 1

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\VMeasure;

$metric = new VMeasure();
```

**References:**

>- A. Rosenberg et al. (2007). V-Measure: A conditional entropy-based external cluster evaluation measure.