<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Reports/ContingencyTable.php">[source]</a></span>

# Contingency Table
A Contingency Table is used to display the frequency distribution of class labels among a clustering. It is similar to a [Confusion Matrix](confusion-matrix.md) but uses the labels to establish ground-truth for a clustering problem instead.

**Estimator Compatibility:** Clusterer

## Parameters
This report does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

$report = new ContingencyTable();

$result = $report->generate($predictions, $labels);

echo $result;
```

```json
[
    {
        "lamb": 11,
        "wolf": 2
    },
    {
        "lamb": 1,
        "wolf": 5
    }
]
```
