<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Reports/ErrorAnalysis.php">[source]</a></span>

# Error Analysis
The Error Analysis report measures the differences between the predicted and target values of a regression problem using multiple error measurements (MAE, MSE, RMSE, MAPE, etc.) as well as statistics regarding the distribution of errors.

**Estimator Compatibility:** Regressor

## Parameters
This report does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Reports\ErrorAnalysis;

$report = new ErrorAnalysis();

$results = $report->generate($predictions, $labels);

echo $results;
```

```json
{
    "mean_absolute_error": 0.18220216502615122,
    "median_absolute_error": 0.17700000000000005,
    "mean_squared_error": 0.05292430893457563,
    "mean_absolute_percentage_error": 18.174348688407402,
    "rms_error": 0.23005283944036775,
    "mean_squared_log_error": 51.96853354084834,
    "r_squared": 0.9999669635675313,
    "error_mean": -0.07112216502615118,
    "error_midrange": -0.12315541256537399,
    "error_median": 0.0007000000000000001,
    "error_variance": 0.04786594657656853,
    "error_mad": 0.17630000000000004,
    "error_iqr": 0.455155412565378,
    "error_skewness": -0.49093461098755187,
    "error_kurtosis": -1.216490935575394,
    "error_min": -0.423310825130748,
    "error_max": 0.17700000000000005,
    "cardinality": 5
}
```