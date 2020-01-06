<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/AggregateReport.php">[source]</a></span>

# Aggregate Report
A report that aggregates the output of multiple reports.

**Estimator Compatibility:** Depends on base reports

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | reports | | array | An array of report objects to aggregate indexed by a user-given name. |

## Example
```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

// Import labels and make predictions

$report = new AggregateReport([
	'breakdown' => new MulticlassBreakdown(),
	'matrix' => new ConfusionMatrix(),
]);

$result = $report->generate($predictions, $labels);
```