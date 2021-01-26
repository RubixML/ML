<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/MeanAbsoluteError.php">[source]</a></span>

# Mean Absolute Error
A scale-dependent metric that measures the average absolute error between a set of predictions and their ground-truth labels. One of the nice properties of MAE is that it has the same units of measurement as the labels being estimated.

$$
{\displaystyle \mathrm {MAE} = {\frac {1}{n}}{\sum _{i=1}^{n}\left |Y_{i}-\hat {Y_{i}}\right|}}
$$

!!! note
    In order to maintain the convention of *maximizing* validation scores, this metric outputs the negative of the original score.

**Estimator Compatibility:** Regressor

**Output Range:** -∞ to 0

## Parameters
This metric does not have any parameters.

## Example
```php
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;

$metric = new MeanAbsoluteError();
```