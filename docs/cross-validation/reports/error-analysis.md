<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/ErrorAnalysis.php">[source]</a></span>

# Residual Analysis
The Error Analysis report measures the differences between the predicted and target values of a regression problem using multiple error measurements (MAE, MSE, RMSE, MAPE, etc.) as well as statistics regarding the distribution of errors.

**Estimator Compatibility:** Regressor

## Parameters
This report does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Reports\ErrorAnalysis;

$report = new ErrorAnalysis();

$result = $report->generate($predictions, $labels);

var_dump($result);
```

```sh
array(18) {
  ["mean_absolute_error"]=> float(0.8)
  ["median_absolute_error"]=> float(1)
  ["mean_squared_error"]=> float(1)
  ["mean_absolute_percentage_error"]=> float(14.020774976657)
  ["rms_error"]=> float(1)
  ["mean_squared_log_error"]=> float(0.019107097505647)
  ["r_squared"]=> float(0.99589305515627)
  ["error_mean"]=> float(-0.2)
  ["error_midrange"]=> float(-0.5)
  ["error_median"]=> float(0)
  ["error_variance"]=> float(0.96)
  ["error_mad"]=> float(1)
  ["error_iqr"]=> float(2)
  ["error_skewness"]=> float(-0.22963966338592)
  ["error_kurtosis"]=> float(-1.0520833333333)
  ["error_min"]=> int(-2)
  ["error_max"]=> int(1)
  ["cardinality"]=> int(10)
}
```