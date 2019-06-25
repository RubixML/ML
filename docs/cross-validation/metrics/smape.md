<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/SMAPE.php">Source</a></span>

# SMAPE
*Symmetric Mean Absolute Percentage Error* expresses the relative error of a set of predictions and their labels as a percentage. It has an upper bound of 100 and a lower bound of 0.

**Estimator Compatibility:** Regressor

**Output Range:** -100 to 0

### Example
```php
use Rubix\ML\CrossValidation\Metrics\SMAPE;

$metric = new SMAPE();
```

### References
>- V. Kreinovich. et al. (2014). How to Estimate Forecasting Quality: A System Motivated Derivation of Symmetric Mean Absolute Percentage Error (SMAPE) and Other Similar Characteristics.