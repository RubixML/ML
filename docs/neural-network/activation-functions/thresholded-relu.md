<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/ThresholdedReLU.php">[source]</a></span>

# Thresholded ReLU
A version of the [ReLU](relu.md) function that activates only if the input is above some user-specified threshold level.

$$
{\displaystyle ThresholdedReLU = {\begin{aligned}&{\begin{cases}0&{\text{if }}x\leq \theta \\x&{\text{if }}x>\theta\end{cases}}\end{aligned}}}
$$

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | threshold | 1.0 | float | The threshold at which the neuron is activated. |

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\ThresholdedReLU;

$activationFunction = new ThresholdedReLU(0.5);
```

## References
[^1]: K. Konda et al. (2015). Zero-bias autoencoders and the benefits of co-adapting features.
