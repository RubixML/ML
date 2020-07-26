# Reports
Reports offer a comprehensive view of the performance of an estimator given the problem in question.

### Generate Report
To generate a report from the predictions of an estimator given the ground truth labels:
```php
public generate(array $predictions, array $labels) : array
```

```php
use Rubix\ML\Reports\ConfusionMatrix;

$predictions = $estimator->predict($dataset);

$report = new ConfusionMatrix();

$result = $report->generate($predictions, $dataset->labels());

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
