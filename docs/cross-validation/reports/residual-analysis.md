<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/ResidualAnalysis.php">Source</a></span>

# Residual Analysis
Residual Analysis is a report that measures the differences between the predicted and target values of a regression problem in detail. In one report, you get multiple error measurements (MAE, MSE, MAPE, R2, etc.) as well as statistics regarding the distribution of errors.

**Estimator Compatibility:** Regressor

### Parameters
This report does not have any parameters.

### Example
```php
use Rubix\ML\CrossValidation\Reports\ResidualAnaysis;

$report = new ResidualAnalysis();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

**Output**

```sh
array(18) {
    ['mean_absolute_error']=> float(0.18220216502615122)
    ['median_absolute_error']=> float(0.17700000000000005)
    ['mean_squared_error']=> float(0.05292430893457563)
    ['mean_absolute_percentage_error']=> float(18.174348688407402)
    ['rms_error']=> float(0.23005283944036775)
    ['mean_squared_log_error']=> float(51.96853354084834)
    ['r_squared']=> float(0.9999669635675313)
    ['error_mean']=> float(-0.07112216502615118)
    ['error_midrange']=> float(-0.12315541256537399)
    ['error_median']=> float(0.0007000000000000001)
    ['error_variance']=> float(0.04786594657656853)
    ['error_mad']=> float(0.17630000000000004)
    ['error_iqr']=> float(0.455155412565378)
    ['error_skewness']=> float(-0.49093461098755187)
    ['error_kurtosis']=> float(-1.216490935575394)
    ['error_min']=> float(-0.423310825130748)
    ['error_max']=> float(0.17700000000000005)
    ['cardinality']=> int(5)
}
```