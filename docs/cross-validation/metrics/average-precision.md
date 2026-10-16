<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/AveragePrecision.php">[source]</a></span>

# Average Precision

Average Precision measures the ability of an estimator to discriminate the positive class from every other class by looking at how well the positive-class probability ranks the samples. It is equivalent to the Area Under the Precision-Recall Curve. A score of 1.0 means that every positive sample has a higher positive-class probability than every negative sample, and a score of 0.0 can only occur when there are no positive samples. When one or more positive samples are present, the minimum score is determined by their prevalence and is strictly greater than 0, so a fully inverted ranking does not score 0.0.

!!! note
    Metric assumes probabilities are between 0 and 1 and their joint distribution sums to 1.

**Estimator Compatibility:** Probabilistic Classifier

**Score Range:** 0 to 1

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | positiveClass | | `string` or `int` | The class considered positive when computing the average precision. |

## Example

```php
use Rubix\ML\CrossValidation\Metrics\AveragePrecision;

$metric = new AveragePrecision('cat');

$probabilities = $estimator->proba($dataset);

$score = $metric->score($probabilities, $dataset->labels());
```
