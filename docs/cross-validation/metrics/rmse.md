### RMSE
The Root Mean Squared Error is equivalent to the average L2 loss.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/RMSE.php)

**Compatibility:** Regression

**Range:** -∞ to 0

**Example:**

```php
use Rubix\ML\CrossValidation\Metrics\RMSE;

$metric = new RMSE();
```