<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Stochastic.php">[source]</a></span>

# Stochastic
A constant learning rate optimizer based on vanilla Stochastic Gradient Descent.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.01 | float | The learning rate that controls the global step size. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;

$optimizer = new Stochastic(0.01);
```