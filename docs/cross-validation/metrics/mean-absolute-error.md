### Mean Absolute Error
A metric that measures the average amount that a prediction is off by given some ground truth (labels).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MeanAbsoluteError.php)

**Compatibility:** Regression

**Range:** -∞ to 0

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

$metric = new MeanAbsoluteError();
```