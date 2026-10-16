<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/ROCAUC.php">[source]</a></span>

# ROC AUC

The Area Under the Receiver Operating Characteristic (ROC) Curve measures the ability of an estimator to discriminate the positive class from every other class by looking at how well the positive-class probability ranks the samples. A score of 1.0 means that every positive sample has a higher positive-class probability than every negative sample, a score of 0.5 means the estimator is random, and a score of 0.0 means the positive and negative samples are perfectly inverted. If either the positive or negative class is entirely absent from the evaluation set the score is undefined and defaults to 0.5 (chance level), which keeps a poorly stratified cross-validation fold from silently dragging down the mean score.

!!! note
    Metric assumes probabilities are between 0 and 1 and their joint distribution sums to 1.

**Estimator Compatibility:** Probabilistic Classifier

**Score Range:** 0 to 1

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | positiveClass | | `string` or `int` | The class considered positive when building the ROC curve. |

## Example

```php
use Rubix\ML\CrossValidation\Metrics\ROCAUC;

$metric = new ROCAUC('cat');

$probabilities = $estimator->proba($dataset);

$score = $metric->score($probabilities, $dataset->labels());
```
