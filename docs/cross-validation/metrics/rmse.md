<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/RMSE.php">Source</a></span>

# RMSE
The Root Mean Squared Error is equivalent to the average L2 loss.

**Estimator Compatibility:** Regressor

**Output Range:** -∞ to 0

### Example
```php
use Rubix\ML\CrossValidation\Metrics\RMSE;

$metric = new RMSE();
```