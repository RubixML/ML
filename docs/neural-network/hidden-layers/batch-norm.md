<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Layers/BatchNorm.php">[source]</a></span>

# Batch Norm
Batch Norm layers normalize the activations of the previous layer such that the mean activation is *close* to 0 and the standard deviation is *close* to 1. Adding Batch Norm reduces the amount of covariate shift within the network which makes it possible to use higher learning rates and thus converge faster under some circumstances.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | decay | 0.9 | float | The decay rate of the previous running averages of the global mean and variance. |
| 2 | betaInitializer | Constant | Initializer | The initializer of the beta parameter. |
| 3 | gammaInitializer | Constant | Initializer | The initializer of the gamma parameter. |

## Example
```php
use Rubix\ML\NeuralNet\Layers\BatchNorm;
use Rubix\ML\NeuralNet\Initializers\Constant;
use Rubix\ML\NeuralNet\Initializers\Normal;

$layer = new BatchNorm(0.7, new Constant(0.), new Normal(1.));
```

## References
[^1]: S. Ioffe et al. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.