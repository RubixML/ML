<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/AdaGrad.php">[source]</a></span>

# AdaGrad
Short for *Adaptive Gradient*, the AdaGrad Optimizer speeds up the learning of parameters that do not change often and slows down the learning of parameters that do enjoy heavy activity. Due to AdaGrad's infinitely decaying step size, training may be slow or fail to converge using a low learning rate.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.01 | float | The learning rate that controls the global step size. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;

$optimizer = new AdaGrad(0.125);
```

## References
[^1]: J. Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.