# Reports
Reports offer a comprehensive view of the performance of an estimator given the problem in question.

### Generate Report
To generate a report from the predictions of an estimator given the ground truth labels:
```php
public generate(array $predictions, array $labels) : array
```

**Example**

```php
use Rubix\ML\Reports\ConfusionMatrix;

// Import labels and make predictions

$report = new ConfusionMatrix();

$result = $report->generate($predictions, $labels);

var_dump($result);
```

```sh
  array(2) {
    ["dog"]=> array(2) {
      ["dog"]=> int(842)
      ["cat"]=> int(5)
    }
    ["cat"]=> array(2) {
      ["dog"]=> int(6)
      ["cat"]=> int(783)
    }
  }
```

### Compatibility
Return a list of integer-encoded estimator types that the report is compatible with:
```php
public compatibility() : array
```

**Example**
```php
var_dump($report->compatibility());
```

```sh
array(2) {
  [0]=> int(1)
  [1]=> int(4)
}
```