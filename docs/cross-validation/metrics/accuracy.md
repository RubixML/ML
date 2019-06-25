<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/Accuracy.php">Source</a></span>

# Accuracy
Accuracy is a quick and simple classification and anomaly detection metric defined as the number of true positives over all samples in the testing set.

**Estimator Compatibility:** Classifier, Anomaly Detector

**Output Range:** 0 to 1

### Example
```php
use Rubix\ML\CrossValidation\Metrics\Accuracy;

$metric = new Accuracy();
```