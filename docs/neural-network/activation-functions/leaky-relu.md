<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/LeakyReLU.php">[source]</a></span>

# Leaky ReLU
Leaky Rectified Linear Units are activation functions that output `x` when x is greater or equal to 0 or `x` scaled by a small *leakage* coefficient when the input is less than 0. Leaky rectifiers have the benefit of allowing a small gradient to flow through during backpropagation even though they might not have activated during the forward pass.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | leakage | 0.1 | float | The amount of leakage as a proportion of the input value to allow to pass through when not inactivated. |

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;

$activationFunction = new LeakyReLU(0.3);
```

### References
>- A. L. Maas et al. (2013). Rectifier Nonlinearities Improve Neural Network Acoustic Models.