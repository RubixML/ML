<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/Softmax/Softmax.php">[source]</a></span>

# Softmax
The Softmax function is a generalization of the [Sigmoid](sigmoid.md) function that squashes each activation between 0 and 1 with the addition that all activations add up to 1. Together, these properties allow the output of the Softmax function to be interpretable as a *joint* probability distribution.

$$
\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

Where:

- 𝑥𝑖 is the i-th element of the input vector
- 𝑛 is the number of elements in the vector
- The denominator ensures the outputs sum to 1

## Parameters
This activation function does not have any parameters.

## Plots
<img src="../../images/activation-functions/softmax.png" alt="Softmax Function" width="500" height="auto">

<img src="../../images/activation-functions/softmax-derivative.png" alt="Softmax Derivative" width="500" height="auto">

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Softmax\Softmax;

$activationFunction = new Softmax();
```
