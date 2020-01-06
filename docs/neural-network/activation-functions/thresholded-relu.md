<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/ThresholdedReLU.php">[source]</a></span>

# Thresholded ReLU
A version of the [ReLU](relu.md) function that activates only if the input is above some user-specified threshold level.

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | threshold | 1.0 | float | The threshold at which the neuron is activated. |

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU;

$activationFunction = new ThresholdedReLU(0.5);
```

### References
>- K. Konda et al. (2015). Zero-bias autoencoders and the benefits of co-adapting features.