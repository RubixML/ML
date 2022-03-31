<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/BrierScore.php">[source]</a></span>

# Brier Score
Brier Score is a *strictly proper* scoring metric that is equivalent to applying mean squared error to the probabilities of a probabilistic estimator.

!!! note
    In order to maintain the convention of *maximizing* validation scores, this metric outputs the negative
    of the original score.

**Estimator Compatibility:** Probabilistic Classifier

**Score Range:** -2 to 0

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\BrierScore;

$metric = new BrierScore();
```

## References
[^1]: G. W. Brier. (1950). Verification of Forecasts Expresses in Terms of Probability.
