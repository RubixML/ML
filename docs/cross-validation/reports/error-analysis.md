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
    "mean absolute error": 0.18220216502615122,
    "median absolute error": 0.17700000000000005,
    "mean squared error": 0.05292430893457563,
    "mean absolute percentage error": 18.174348688407402,
    "rms error": 0.23005283944036775,
    "mean squared log error": 51.96853354084834,
    "r squared": 0.9999669635675313,
    "error mean": -0.07112216502615118,
    "error median": 0.0007000000000000001,
    "error variance": 0.04786594657656853,
    "error stddev": 0.2187828754189151,
    "error mad": 0.17630000000000004,
    "error iqr": 0.455155412565378,
    "error skewness": -0.49093461098755187,
    "error kurtosis": -1.216490935575394,
    "error min": -0.423310825130748,
    "error max": 0.17700000000000005,
    "cardinality": 5
}
```
