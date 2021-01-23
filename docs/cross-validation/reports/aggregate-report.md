<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Reports/AggregateReport.php">[source]</a></span>

# Aggregate Report
A report generator that aggregates the output of multiple reports.

**Estimator Compatibility:** Depends on base reports

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | reports | | array | An array of report generators to aggregate keyed by a user-specified name. |

## Example
```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new AggregateReport([
	'breakdown' => new MulticlassBreakdown(),
	'matrix' => new ConfusionMatrix(),
]);
```