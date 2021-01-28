<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/RMSE.php">[source]</a></span>

# RMSE
The Root Mean Squared Error (RMSE) is equivalent to the standard deviation of the error residuals in a regression problem. Since RMSE is just the square root of the [MSE](mean-squared-error.md), RMSE is also sensitive to outliers because larger errors have a disproportionately large effect on the score.

$$
{\displaystyle \operatorname {RMSE} = {\sqrt{ \frac {1}{n} \sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}}}}
$$

!!! note
    In order to maintain the convention of *maximizing* validation scores, this metric outputs the negative of the original score.

**Estimator Compatibility:** Regressor

**Output Range:** -∞ to 0

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\RMSE;

$metric = new RMSE();
```