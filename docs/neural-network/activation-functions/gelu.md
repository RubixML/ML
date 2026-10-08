<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/GELU.php">[source]</a></span>

# GELU

Gaussian Error Linear Units (GELUs) are rectifiers that are gated by the magnitude of their input rather than the sign of their input as with ReLU variants. Their output can be interpreted as the expected value of a neuron with random dropout regularization applied.

$$
{\displaystyle \operatorname{GELU}(x) = x \, \Phi(x) = \frac{x}{2}\left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)}
$$

$$
{\displaystyle \operatorname{GELU}'(x) = \Phi(x) + x \, \varphi(x) = \frac{1}{2}\left(1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right) + \frac{x}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)}
$$

## Parameters

This activation function does not have any parameters.

## Example

```php
use Rubix\ML\NeuralNet\ActivationFunctions\GELU;

$activationFunction = new GELU();
```

### References

>- D. Hendrycks et al. (2018). Gaussian Error Linear Units (GELUs).
