<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/ProbabilisticAccuracy.php">[source]</a></span>

# Probabilistic Accuracy
This metric comes from the sports betting domain, where it's used to measure the accuracy of predictions by looking at the probabilities of class predictions. Accordingly, this metric places additional weight on the "confidence" of each prediction.

!!! note
    Metric assumes probabilities are between 0 and 1 and their joint distribution sums to 1.

**Estimator Compatibility:** Probabilistic Classifier

**Output Range:** 0 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\ProbabilisticAccuracy;

$metric = new ProbabilisticAccuracy();
```

## References
[^1]: https://mercurius.io/en/learn/predicting-forecasting-football
