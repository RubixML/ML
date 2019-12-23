<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/ContingencyTable.php">[source]</a></span>

# Contingency Table
A Contingency Table is used to display the frequency distribution of class labels among a clustering. It is similar to a [Confusion Matrix](confusion-matrix.md) but uses the labels to establish ground-truth for a clustering problem instead.

**Estimator Compatibility:** Clusterer

## Parameters
This report does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

// Import labels and make predictions

$report = new ContingencyTable();

$result = $report->generate($predictions, $labels);

var_dump($result);
```

```sh
array(3) {
  [0]=>
    array(3) {
      ["dog"]=> int(13)
      ["frog"]=> int(0)
      ["cat"]=> int(2)
    }
  [1]=>
    array(3) {
      ["dog"]=> int(1)
      ["frog"]=> int(0)
      ["cat"]=> int(12)
    }
  [2]=>
    array(3) {
      ["dog"]=> int(0)
      ["frog"]=> int(14)
      ["cat"]=> int(0)
    }
  }
```