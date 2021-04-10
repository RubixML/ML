<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Reports/MulticlassBreakdown.php">[source]</a></span>

# Multi-class Breakdown
A multiclass classification report that computes a number of metrics (Accuracy, Precision, Recall, etc.) derived from their confusion matrix on an overall and individual class basis.

**Estimator Compatibility:** Classifier, Anomaly Detector

## Parameters
This report does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;

$report = new MulticlassBreakdown();

$results = $report->generate($predictions, $labels);

echo $results;
```

```json
{
    "overall": {
        "accuracy": 0.6,
        "accuracy balanced": 0.5833333333333333,
        "f1 score": 0.5833333333333333,
        "precision": 0.5833333333333333,
        "recall": 0.5833333333333333,
        "specificity": 0.5833333333333333,
        "negative predictive value": 0.5833333333333333,
        "false discovery rate": 0.4166666666666667,
        "miss rate": 0.4166666666666667,
        "fall out": 0.4166666666666667,
        "false omission rate": 0.4166666666666667,
        "mcc": 0.16666666666666666,
        "informedness": 0.16666666666666652,
        "markedness": 0.16666666666666652,
        "true positives": 3,
        "true negatives": 3,
        "false positives": 2,
        "false negatives": 2,
        "cardinality": 5
    },
    "classes": {
        "wolf": {
            "accuracy": 0.6,
            "accuracy balanced": 0.5833333333333333,
            "f1 score": 0.6666666666666666,
            "precision": 0.6666666666666666,
            "recall": 0.6666666666666666,
            "specificity": 0.5,
            "negative predictive value": 0.5,
            "false discovery rate": 0.33333333333333337,
            "miss rate": 0.33333333333333337,
            "fall out": 0.5,
            "false omission rate": 0.5,
            "informedness": 0.16666666666666652,
            "markedness": 0.16666666666666652,
            "mcc": 0.16666666666666666,
            "true positives": 2,
            "true negatives": 1,
            "false positives": 1,
            "false negatives": 1,
            "cardinality": 3,
            "proportion": 0.6
        },
        "lamb": {
            "accuracy": 0.6,
            "accuracy balanced": 0.5833333333333333,
            "f1 score": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "specificity": 0.6666666666666666,
            "negative predictive value": 0.6666666666666666,
            "false discovery rate": 0.5,
            "miss rate": 0.5,
            "fall out": 0.33333333333333337,
            "false omission rate": 0.33333333333333337,
            "informedness": 0.16666666666666652,
            "markedness": 0.16666666666666652,
            "mcc": 0.16666666666666666,
            "true positives": 1,
            "true negatives": 2,
            "false positives": 1,
            "false negatives": 1,
            "cardinality": 2,
            "proportion": 0.4
        }
    }
}
```
