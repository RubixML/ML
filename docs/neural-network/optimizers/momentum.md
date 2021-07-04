<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Momentum.php">[source]</a></span>

# Momentum
Momentum accelerates each update step by accumulating velocity from past updates and adding a factor of the previous velocity to the current step. Momentum can help speed up training and escape bad local minima when compared with [Stochastic](stochastic.md) Gradient Descent.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate that controls the global step size. |
| 2 | decay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | lookahead | false | bool | Should we employ Nesterov's lookahead (NAG) when updating the parameters? |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\Momentum;

$optimizer = new Momentum(0.01, 0.1, true);
```

## References
[^1]: D. E. Rumelhart et al. (1988). Learning representations by back-propagating errors.
[^2]: I. Sutskever et al. (2013). On the importance of initialization and momentum in deep learning.
