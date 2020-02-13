<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/MulticlassBreakdown.php">[source]</a></span>

# Multi-class Breakdown
A classification and anomaly detection report that drills down into unique class statistics as well as provide an overall picture. The report includes metrics such as Accuracy, F1 Score, MCC, Precision, Recall, Fall Out, and Miss Rate.

**Estimator Compatibility:** Classifier, Anomaly Detector

## Parameters
This report does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new MulticlassBreakdown();

$result = $report->generate($predictions, $labels);

var_dump($result);
```

```sh
...
['classes']=> array(2) {
	['wolf']=> array(19) {
      	['accuracy']=> float(0.6)
		['accuracy_balanced']=> float(0.5833333333333333)
		['f1_score']=> float(0.66666666666667)
      	['precision']=> float(0.66666666666667)
      	['recall']=> float(0.66666666666667)
      	['specificity']=> float(0.5)
      	['negative_predictive_value']=> float(0.5)
      	['false_discovery_rate']=> float(0.33333333333333)
      	['miss_rate']=> float(0.33333333333333)
      	['fall_out']=> float(0.5)
      	['false_omission_rate']=> float(0.5)
		['threat_score']=> float(0.5)
      	['mcc']=> float(0.16666666666667)
      	['informedness']=> float(0.16666666666667)
      	['markedness']=> float(0.16666666666667)
      	['true_positives']=> int(2)
      	['true_negatives']=> int(1)
      	['false_positives']=> int(1)
      	['false_negatives']=> int(1)
      	['cardinality']=> int(3)
      	['percentage']=> float(60.0)
    }
...
```