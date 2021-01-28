<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/Sigmoid.php">[source]</a></span>

# Sigmoid
A bounded S-shaped function (sometimes called the *Logistic* function) with an output value between 0 and 1. The output of the sigmoid function has the advantage of being interpretable as a probability, however it is not zero-centered and tends to saturate if inputs become large.

$$
{\displaystyle Sigmoid = {\frac {1}{1+e^{-x}}}}
$$

## Parameters
This activation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;

$activationFunction = new Sigmoid();
```