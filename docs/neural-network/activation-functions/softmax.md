<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/Softmax.php">[source]</a></span>

# Softmax
The Softmax function is a generalization of the [Sigmoid](sigmoid.md) function that squashes each activation between 0 and 1 with the addition that all activations add up to 1. Together, these properties allow the output of the Softmax function to be interpretable as a *joint* probability distribution.

$$
{\displaystyle Softmax = {\frac {e^{x_{i}}}{\sum _{j=1}^{J}e^{x_{j}}}}}
$$

## Parameters
This activation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax;

$activationFunction = new Softmax();
```