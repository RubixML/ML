<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/RMSProp.php">[source]</a></span>

# RMS Prop
An adaptive gradient technique that divides the current gradient over a rolling window of the magnitudes of recent gradients. Unlike [AdaGrad](adagrad.md), RMS Prop does not suffer from an infinitely decaying step size.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | rate | 0.001 | float | The learning rate that controls the global step size. |
| 2 | decay | 0.1 | float | The decay rate of the rms property. |

## Example
```php
use Rubix\ML\NeuralNet\Optimizers\RMSProp;

$optimizer = new RMSProp(0.01, 0.1);
```

## References
[^1]: T. Tieleman et al. (2012). Lecture 6e rmsprop: Divide the gradient by a running average of its recent magnitude.