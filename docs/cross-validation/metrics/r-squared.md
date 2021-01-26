<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/RSquared.php">[source]</a></span>

# R Squared
The *coefficient of determination* or R Squared (R²) is the proportion of the variance in the target labels that is explainable from the predictions. It gives an indication as to how well the predictions approximate the labels.

$$
{\displaystyle R^{2} = 1-{SS_{\rm {res}} \over SS_{\rm {tot}}}}
$$

**Estimator Compatibility:** Regressor

**Output Range:** -∞ to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\RSquared;

$metric = new RSquared();
```
