<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/MeanSquaredError.php">[source]</a></span>

# Mean Squared Error
A scale-dependent regression metric that gives greater weight to error scores the worse they are. Formally, Mean Squared Error (MSE) is the average of the squared differences between a set of predictions and their target labels.

$$
{\displaystyle \operatorname {MSE} = {\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}}
$$

!!! note
    In order to maintain the convention of *maximizing* validation scores, this metric outputs the negative of the original score.

**Estimator Compatibility:** Regressor

**Output Range:** -∞ to 0

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;

$metric = new MeanSquaredError();
```