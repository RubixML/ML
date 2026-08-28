<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/SoftPlus/SoftPlus.php">[source]</a></span>

# SoftPlus
A smooth approximation of the piecewise linear [ReLU](relu.md) activation function.

$$
{\displaystyle SoftPlus = \log \left(1+e^{x}\right)}
$$

## Parameters
This activation function does not have any parameters.

## Plots
<img src="../../images/activation-functions/SoftPlus.png" alt="SoftPlus Function" width="500" height="auto">

<img src="../../images/activation-functions/SoftPlus-derivative.png" alt="SoftPlus Derivative" width="500" height="auto">

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SoftPlus\SoftPlus;

$activationFunction = new SoftPlus();
```

## References
[1]: X. Glorot et al. (2011). Deep Sparse Rectifier Neural Networks.
