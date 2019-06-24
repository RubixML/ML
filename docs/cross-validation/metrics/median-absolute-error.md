### Median Absolute Error
Median Absolute Error (MAE) is a robust measure of the error that ignores highly erroneous predictions.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MedianAbsoluteError.php)

**Compatibility:** Regression

**Range:** -∞ to 0

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\MedianAbsoluteError;

$metric = new MedianAbsoluteError();
```