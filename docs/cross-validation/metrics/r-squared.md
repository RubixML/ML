<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/RSquared.php">Source</a></span>

# R Squared
The *coefficient of determination* or R Squared (R²) is the proportion of the variance in the dependent variable that is predictable from the independent variable(s).

**Estimator Compatibility:** Regressor

**Output Range:** -∞ to 1

### Example
```php
use Rubix\ML\CrossValidation\Metrics\RSquared;

$metric = new RSquared();
```