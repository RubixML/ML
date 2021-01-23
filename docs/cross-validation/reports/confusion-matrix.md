<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Reports/ConfusionMatrix.php">[source]</a></span>

# Confusion Matrix
A Confusion Matrix is a square matrix (table) that visualizes the true positives, false positives, true negatives, and false negatives of a set of predictions and their corresponding labels.

**Estimator Compatibility:** Classifier, Anomaly Detector

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | classes| | array | The classes to include in the report. If null then all classes will be included. |

## Example
```php
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;

$report = new ConfusionMatrix([
  'dog', 'cat', 'turtle',
]);

$result = $report->generate($predictions, $labels);

echo $result;
```

```json
{
    "dog": {
        "dog": 12,
        "cat": 3,
        "turtle": 0
    },
    "cat": {
        "dog": 2,
        "cat": 9,
        "turtle": 1
    },
    "turtle": {
        "dog": 1,
        "cat": 0,
        "turtle": 11
    }
}
```