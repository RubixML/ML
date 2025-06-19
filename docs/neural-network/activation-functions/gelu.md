<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/GELU/GELU.php">[source]</a></span>

# GELU
Gaussian Error Linear Units (GELUs) are rectifiers that are gated by the magnitude of their input rather than the sign of their input as with ReLU variants. Their output can be interpreted as the expected value of a neuron with random dropout regularization applied.

$$
\displaystyle
\text{GELU}(x) = \frac{x}{2} \left[1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

## Parameters
This activation function does not have any parameters.

## Size and Performance
GELU is computationally more expensive than simpler activation functions like ReLU due to its use of hyperbolic tangent and exponential calculations. The implementation uses an approximation formula to improve performance, but it still requires more computational resources. Despite this cost, GELU has gained popularity in transformer architectures and other deep learning models due to its favorable properties for training deep networks.

## Plots
<img src="../../images/activation-functions/gelu.png" alt="GELU Function" width="500" height="auto">

<img src="../../images/activation-functions/gelu-derivative.png" alt="GELU Derivative" width="500" height="auto">

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\GELU;

$activationFunction = new GELU();
```

### References
>- D. Hendrycks et al. (2018). Gaussian Error Linear Units (GELUs).
