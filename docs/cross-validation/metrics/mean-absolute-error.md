<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MeanAbsoluteError.php">Source</a></span></p>

# Mean Absolute Error
A metric that measures the average amount that a prediction is off by given some ground truth (labels).

**Estimator Compatibility:** Regressor

**Output Range:** -∞ to 0

### Example
```php
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

$metric = new MeanAbsoluteError();
```