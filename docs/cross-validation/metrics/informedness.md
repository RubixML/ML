<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/Informedness.php">[source]</a></span>

# Informedness
Informedness a multiclass generalization of Youden's J Statistic and can be interpreted as the probability that an estimator will make an informed prediction. Its value ranges from -1 through 1 and has a value of 0 when the test yields no useful information.

$$
{\displaystyle Informedness = {\frac {\text{TP}}{{\text{TP}}+{\text{FN}}}}+{\frac {\text{TP}}{{\text{TN}}+{\text{FP}}}}-1}
$$

**Estimator Compatibility:** Classifier, Anomaly Detector

**Output Range:** -1 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\Informedness;

$metric = new Informedness();
```

## References
[^1]: W. J. Youden. (1950). Index for Rating Diagnostic Tests.