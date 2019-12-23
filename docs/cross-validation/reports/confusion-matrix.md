<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/ConfusionMatrix.php">[source]</a></span>

# Confusion Matrix
A Confusion Matrix is a table that visualizes the true positives, false positives, true negatives, and false negatives of a classification experiment. The name stems from the fact that the matrix makes it clear to see if the estimator is *confusing* any two classes.

**Estimator Compatibility:** Classifier, Anomaly Detector

## Parameters
This report does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;

// Import labels and make predictions

$report = new ConfusionMatrix();

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