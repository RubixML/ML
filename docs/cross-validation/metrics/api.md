# Metrics
Validation metrics are for evaluating the performance of an Estimator. They output a score based on the predictions and the ground-truth labels.

### Scoring Predictions

To compute a validation score, pass in the predictions from an estimator along with the expected labels:
```php
public score(array $predictions, array $labels) : float
```

**Example**

```php
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

// Train an estimator and make predictions

$metric = new MeanAbsoluteError();

$score = $metric->score($predictions, $labels);

var_dump($score);
```

**Output**

```sh
float(-0.99846070553066)
```

> **Note:** Regression metrics output the negative of their value to maintain the notion that cross validation scores should be *maximized* instead of *minimized* such as the case with loss functions.

### Output Range
Output the range of values the validation score can take on in a [2-tuple](../../faq.md#what-is-a-tuple):
```php
public range() : array
```

**Example**

```php
[$min, $max] = $metric->range();

var_dump($min);
var_dump($max);
```

**Output**

```sh
float(-INF)

int(0)
```

### Compatibility
Return a list of integer-encoded estimator types that the metric is compatible with:
```php
public compatibility() : array
```

**Example**
```php
var_dump($metric->compatibility());
```

**Output**

```sh
array(1) {
  [0]=> int(2)
}
```