<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/SMAPE.php">[source]</a></span>

# SMAPE
*Symmetric Mean Absolute Percentage Error* (SMAPE) is a scale-independent regression metric that expresses the relative error of a set of predictions and their labels as a percentage. It is an improvement over the non-symmetric MAPE in that it is both upper and lower bounded.

$$
{\displaystyle {\text{SMAPE}} = {\frac {100\%}{n}}\sum _{t=1}^{n}{\frac {\left|F_{t}-A_{t}\right|}{(|A_{t}|+|F_{t}|)/2}}}
$$

!!! note
    In order to maintain the convention of *maximizing* validation scores, this metric outputs the negative of the original score.

**Estimator Compatibility:** Regressor

**Score Range:** -100 to 0

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\SMAPE;

$metric = new SMAPE();
```

## References
[^1]: V. Kreinovich. et al. (2014). How to Estimate Forecasting Quality: A System Motivated Derivation of Symmetric Mean Absolute Percentage Error (SMAPE) and Other Similar Characteristics.