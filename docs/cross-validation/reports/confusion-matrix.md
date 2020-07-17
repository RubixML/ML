<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/ConfusionMatrix.php">[source]</a></span>

# Confusion Matrix
A Confusion Matrix is a square matrix (table) that visualizes the true positives, false positives, true negatives, and false negatives of a set of predictions and their corresponding labels.

**Estimator Compatibility:** Classifier, Anomaly Detector

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | classes| | array | The classes to include in the report. If null then all classes will be included. |

## Example
```php
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;

$report = new ConfusionMatrix([
  'dog', 'cat', 'turtle',
]);

$result = $report->generate($predictions, $labels);

var_dump($result);
```

```sh
array(3) {
  ["dog"]=> array(3) {
    ["dog"]=> int(842)
    ["cat"]=> int(5)
    ["turtle"]=> int(0)
  }
  ["cat"]=> array(3) {
    ["dog"]=> int(0)
    ["cat"]=> int(783)
    ["turtle"]=> int(3)
  }
  ["turtle"]=> array(2) {
    ["dog"]=> int(31)
    ["cat"]=> int(79)
    ["turtle"]=> int(496)
  }
}
```