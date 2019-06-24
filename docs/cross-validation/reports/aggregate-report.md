<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/AggregateReport.php">Source</a></span></p>

# Aggregate Report
A report that aggregates the results of multiple reports. The reports are indexed by their key given at construction time.

**Estimator Compatibility:** Depends on base reports

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | reports | | array | An array of report objects to aggregate. |

### Example
```php
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new AggregateReport([
	'breakdown' => new MulticlassBreakdown(),
	'matrix' => new ConfusionMatrix(),
]);

$result = $report->generate($estimator, $testing);
```