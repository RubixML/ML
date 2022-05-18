# Metrics
Validation metrics are for used evaluating the generalization performance of an estimator. They output a score based on the predictions and known ground-truth labels.

!!! note
    Some regression metrics output the negative of their value to maintain the convention that scores get better as they *increase*.

### Scoring Predictions
To compute a validation score, pass in the predictions from an estimator along with their expected labels.

```php
public score(array $predictions, array $labels) : float
```

```php
use Rubix\ML\CrossValidation\Metrics\FBeta;

$predictions = $estimator->predict($dataset);

$metric = new FBeta(1.0);

$score = $metric->score($predictions, $dataset->labels());

echo $score;
```

```
0.88
```

### Scoring Probabilities
Metrics that implement the ProbabilisticMetric interface calculate a validation score derived from the estimated probabilities of a [Probabilistic](../../probabilistic.md) estimator and their corresponding ground-truth labels.

```php
public score(array $probabilities, array $labels) : float
```

!!! note
    Metric assumes probabilities are values between 0 and 1 and their joint distribution sums to exactly 1 for each sample.

```php
use Rubix\ML\CrossValidation\Metrics\ProbabilisticAccuracy;

$probabilities = $estimator->proba($dataset);

$metric = new ProbabilisticAccuracy;

$score = $metric->score($probabilities, $dataset->labels());
```

### Score Range
Output the minimum and maximum value the validation score can take in a [2-tuple](../../faq.md#what-is-a-tuple).

```php
public range() : Rubix\ML\Tuple{float, float}
```

```php
[$min, $max] = $metric->range()->list();

echo "min: $min, max: $max";
```

```
min: 0.0, max: 1.0
```
