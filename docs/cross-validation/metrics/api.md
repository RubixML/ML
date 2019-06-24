### Validation Metrics
Validation metrics are for evaluating the performance of an Estimator. They output a score based on the predictions and the ground-truth labels.

> **Note**: Regression metrics output the negative of their value to maintain the notion that cross validation scores should be *maximized* instead of *minimized* such as the case with loss functions.

To compute a validation score, pass in the predictions from an estimator along with the expected labels:
```php
public score(array $predictions, array $labels) : float
```

Output the range of values the validation score can take on in a 2-tuple:
```php
public range() : array
```

Return a list of estimators that metric is compatible with:
```php
public compatibility() : array
```

### Example
```php
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

$metric = new MeanAbsoluteError();

$score = $metric->score($predictions, $labels);

var_dump($metric->range());

var_dump($score);
```

**Output:**
```sh
array(2) {
  [0]=> float(-INF)
  [1]=> int(0)
}

float(-0.99846070553066)
```