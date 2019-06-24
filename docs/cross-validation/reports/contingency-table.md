<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Reports/ContingencyTable.php">Source</a></span></p>

# Contingency Table
A Contingency Table is used to display the frequency distribution of class labels among a clustering of samples. It is similar to a [Confusion Matrix](#confusion-matrix) but uses the labels to establish a ground truth for a clustering instead.

**Estimator Compatibility:** Clusterer

### Parameters
This report does not have any parameters.

### Example
```php
use Rubix\ML\CrossValidation\Reports\ContingencyTable;

$report = new ContingencyTable();

$result = $report->generate($estimator, $testing);

var_dump($result);
```

**Output:**

```sh
array(3) {
    [1]=>
    array(3) {
      [1]=> int(13)
      [2]=> int(0)
      [3]=> int(2)
    }
    [2]=>
    array(3) {
      [1]=> int(1)
      [2]=> int(0)
      [3]=> int(12)
    }
    [0]=>
    array(3) {
      [1]=> int(0)
      [2]=> int(14)
      [3]=> int(0)
    }
  }
```