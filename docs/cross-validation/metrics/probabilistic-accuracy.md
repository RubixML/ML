<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/ProbabilisticAccuracy.php">[source]</a></span>

# Probabilistic Accuracy
This metric comes from the sports betting domain, where it's used to measure the accuracy of predictions by looking at the probabilities of class predictions. Accordingly, this metric places additional weight on the "confidence" of each prediction.

**Estimator Compatibility:** Probabilistic Classifier

**Score Range:** 0 to 1

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\ProbabilisticAccuracy;

$metric = new ProbabilisticAccuracy();
```

## References
[^1]: https://mercurius.io/en/learn/predicting-forecasting-football
