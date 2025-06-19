<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/Softplus/Softplus.php">[source]</a></span>

# Softplus
A smooth approximation of the piecewise linear [ReLU](relu.md) activation function.

$$
{\displaystyle Softplus = \log \left(1+e^{x}\right)}
$$

## Parameters
This activation function does not have any parameters.

## Size and Performance
Softplus is computationally more expensive than ReLU due to its use of both exponential and logarithmic calculations. Each activation requires computing an exponential term, an addition, and a logarithm. This makes Softplus significantly more resource-intensive than simpler activation functions, especially in large networks. However, Softplus provides a smooth, differentiable alternative to ReLU with no zero-gradient regions, which can improve gradient flow during training for certain types of networks. The trade-off between computational cost and the benefits of smoothness should be considered when choosing between Softplus and ReLU.

## Plots
<img src="../../images/activation-functions/softplus.png" alt="Softplus Function" width="500" height="auto">

<img src="../../images/activation-functions/softplus-derivative.png" alt="Softplus Derivative" width="500" height="auto">

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Softplus;

$activationFunction = new Softplus();
```

## References
[^1]: X. Glorot et al. (2011). Deep Sparse Rectifier Neural Networks.
