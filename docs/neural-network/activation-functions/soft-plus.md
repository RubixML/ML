<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/SoftPlus.php">[source]</a></span>

# Soft Plus
A smooth approximation of the piecewise linear [ReLU](relu.md) activation function.

## Parameters
This activation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus;

$activationFunction = new SoftPlus();
```

### References
>- X. Glorot et al. (2011). Deep Sparse Rectifier Neural Networks.