<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/SMAPE.php">[source]</a></span>

# SMAPE
*Symmetric Mean Absolute Percentage Error* (SMAPE) is a scale-independent regression metric that expresses the relative error of a set of predictions and their labels as a percentage. It is an improvement over the non-symmetric MAPE in that it is both upper and lower bounded.

> **Note:** In order to maintain the convention of *maximizing* validation scores, this metric outputs the negative of the original score.

**Estimator Compatibility:** Regressor

**Output Range:** -100 to 0

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\SMAPE;

$metric = new SMAPE();
```

### References
>- V. Kreinovich. et al. (2014). How to Estimate Forecasting Quality: A System Motivated Derivation of Symmetric Mean Absolute Percentage Error (SMAPE) and Other Similar Characteristics.