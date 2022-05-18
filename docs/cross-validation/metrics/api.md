# Metrics
Validation metrics are for used evaluating the performance of an estimator. They output a score based on the predictions and the ground-truth found in the labels.

### Scoring Predictions
To compute a validation score, pass in the predictions from an estimator along with the expected labels:
```php
public score(array $predictions, array $labels) : float
```

```php
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

$predictions = $estimator->predict($dataset);

$metric = new MeanAbsoluteError();

$score = $metric->score($predictions, $dataset->labels());

echo $score;
```

```
-0.99846
```

!!! note
    Regression metrics output the negative of their value to maintain the notion that cross validation scores should be *maximized* instead of *minimized* such as the case with loss functions.

### Output Range
Output the minimum and maximum value the validation score can take in a [2-tuple](../../faq.md#what-is-a-tuple):
```php
public range() : Rubix\ML\Tuple{float, float}
```

```php
[$min, $max] = $metric->range()->list();

echo "min: $min, max: $max";
```

```
min: -INF, max: 0
```
