<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/TopKAccuracy.php">[source]</a></span>

# Top K Accuracy
Top K Accuracy looks at the k classes with the highest predicted probabilities when calculating the accuracy score. If one of the top k classes matches the ground-truth, then the prediction is considered accurate.

**Estimator Compatibility:** Probabilistic Classifier

**Score Range:** 0 to 1

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 3 | int | The number of classes with the highest predicted probability to consider. |

## Example
```php
use Rubix\ML\CrossValidation\Metrics\TopKAccuracy;

$metric = new TopKAccuracy(5);
```
