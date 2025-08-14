<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/SiLU/SiLU.php">[source]</a></span>

# SiLU
Sigmoid Linear Units are smooth and non-monotonic rectified activation functions. Their inputs are weighted by the [Sigmoid](sigmoid.md) activation function acting as a self-gating mechanism.

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

Where
- σ(x) is the sigmoid function.

## Parameters
This activation function does not have any parameters.

## Plots
<img src="../../images/activation-functions/silu.png" alt="SiLU Function" width="500" height="auto">

<img src="../../images/activation-functions/silu-derivative.png" alt="SiLU Derivative" width="500" height="auto">

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU\SiLU;

$activationFunction = new SiLU();
```

## References
[1]: S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning.
