<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/SoftPlus.php">[source]</a></span>

# Soft Plus
A smooth approximation of the piecewise linear [ReLU](relu.md) activation function.

$$
{\displaystyle Soft-Plus = \log \left(1+e^{x}\right)}
$$

## Parameters
This activation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;

$activationFunction = new SoftPlus();
```

## References
[^1]: X. Glorot et al. (2011). Deep Sparse Rectifier Neural Networks.