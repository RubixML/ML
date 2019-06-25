<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/Optimizers/Stochastic.php">Source</a></span>

# Stochastic
A constant learning rate optimizer based on the original Stochastic Gradient Descent paper.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.01 | float | The learning rate. i.e. the global step size. |

### Example
```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;

$optimizer = new Stochastic(0.01);
```