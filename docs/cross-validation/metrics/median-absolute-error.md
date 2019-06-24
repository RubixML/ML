<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MedianAbsoluteError.php">Source</a></span></p>

# Median Absolute Error
Median Absolute Error (MAE) is a robust measure of the error that ignores highly erroneous predictions.

**Estimator Compatibility:** Regressor

**Output Range:** -∞ to 0

### Example
```php
use Rubix\ML\CrossValidation\Metrics\MedianAbsoluteError;

$metric = new MedianAbsoluteError();
```