### Reports
Reports offer a comprehensive view of the performance of an estimator given the problem in question.

To generate a report from the predictions of an estimator given some ground truth labels:
```php
public generate(array $predictions, array $labels) : array
```

Return a list of estimators that report is compatible with:
```php
public compatibility() : array
```

**Example:**

```php
use Rubix\ML\Reports\ConfusionMatrix;

$report = new ConfusionMatrix();

$result = $report->generate($predictions, $labels);
```