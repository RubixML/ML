<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/Swish.php">[source]</a></span>

# Swish
Swish is a smooth and non-monotonic rectified activation function. The inputs are weighted by the [Sigmoid](sigmoid.md) activation function acting as a self-gating mechanism. In addition, the `beta` parameter allows you to adjust the gate such that you can interpolate between the scaled linear function and ReLU as `beta` goes from 0 to infinity. When `beta` is equal to 1, Swish is equivalent to the Sigmoid-weighted Linear Unit or *SiLU*.

## Parameters
## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | beta | 1.0 | float | The parameter that adjusts the slope of the sigmoid gating mechanism. |

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Swish;

$activationFunction = new Swish(1.0);
```

### References
>- S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning.
>- P. Ramachandran er al. (2017). Swish: A Self-gated Activation Function.
>- P. Ramachandran et al. (2017). Searching for Activation Functions.
