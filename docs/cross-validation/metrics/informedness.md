<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/Informedness.php">Source</a></span>

# Informedness
Informedness is a measure of the probability that an estimator will make an informed decision. The index was suggested by W.J. Youden as a way of summarizing the performance of a diagnostic test. Its value ranges from -1 through 1 and has a zero value when the test gives yields no useful information.

**Estimator Compatibility:** Classifier, Anomaly Detector

**Output Range:** -1 to 1

### Example
```php
use Rubix\ML\CrossValidation\Metrics\Informedness;

$metric = new Informedness();
```

### References
>- W. J. Youden. (1950). Index for Rating Diagnostic Tests.