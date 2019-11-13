<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/Metrics/MeanSquaredError.php">[source]</a></span>

# Mean Squared Error
A scale-dependent regression metric that punishes bad predictions more the worse they are. Formally, MSE is the average of the squared differences between a set of predictions and their target labels. For an unbiased estimator, the MSE measures the variance of the predictions.

> **Note:** In order to maintain the convention of *maximizing* validation scores, this metric outputs the negative of the original score.

**Estimator Compatibility:** Regressor

**Output Range:** -∞ to 0

### Example
```php
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;

$metric = new MeanSquaredError();
```