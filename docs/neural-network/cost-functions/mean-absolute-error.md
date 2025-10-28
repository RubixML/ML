<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/CostFunctions/MeanAbsoluteError/MeanAbsoluteError.php">[source]</a></span>

# Mean Absolute Error
Mean Absolute Error (MAE) measures the average magnitude of errors between predicted and actual values without considering their direction. It is a linear score which means all individual differences are weighted equally. MAE is more robust to outliers compared to Mean Squared Error (MSE) because it doesn't square the differences.

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$

## Parameters
This cost function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\CostFunctions\MeanAbsoluteError\MeanAbsoluteError;

$costFunction = new MeanAbsoluteError();
```
