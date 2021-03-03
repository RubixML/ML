<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/AdaMax.php">[source]</a></span>

# AdaMax
A version of the [Adam](adam.md) optimizer that replaces the RMS property with the infinity norm of the past gradients. As such, AdaMax is generally more suitable for sparse parameter updates and noisy gradients.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate that controls the global step size. |
| 2 | momentumDecay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | normDecay | 0.001 | float | The decay rate of the infinity norm. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\AdaMax;

$optimizer = new AdaMax(0.0001, 0.1, 0.001);
```

## References
[^1]: D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.