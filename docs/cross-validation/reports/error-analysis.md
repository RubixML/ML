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
    "mean absolute error": 0.8,
    "median absolute error": 1,
    "mean squared error": 1,
    "mean absolute percentage error": 14.02077497665733,
    "rms error": 1,
    "mean squared log error": 0.019107097505647368,
    "r squared": 0.9958930551562692,
    "error mean": -0.2,
    "error standard deviation": 0.9898464007663,
    "error skewness": -0.22963966338592326,
    "error kurtosis": -1.0520833333333324,
    "error min": -2,
    "error 25%": -1.0,
    "error median": 0.0,
    "error 75%": 0.75,
    "error max": 1,
    "cardinality": 10
}
```
