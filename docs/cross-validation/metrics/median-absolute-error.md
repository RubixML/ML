<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/MedianAbsoluteError.php">[source]</a></span>

# Median Absolute Error
Median Absolute Error (MAD) is a robust measure of error, similar to [MAE](mean-absolute-error.md), that ignores highly erroneous predictions. Since MAD is a robust statistic, it works well even when used to measure non-normal distributions.

$$
{\displaystyle \operatorname {MAD} = \operatorname {median} (|Y_{i}-{\tilde {Y}}|)}
$$

!!! note
    In order to maintain the convention of *maximizing* validation scores, this metric outputs the negative of the original score.

**Estimator Compatibility:** Regressor

**Score Range:** -∞ to 0

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\MedianAbsoluteError;

$metric = new MedianAbsoluteError();
```
