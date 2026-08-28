<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/SELU/SELU.php">[source]</a></span>

# SELU
Scaled Exponential Linear Units (SELU) are a self-normalizing activation function based on the [ELU](#elu) activation function. Neuronal activations of SELU networks automatically converge toward zero mean and unit variance, unlike explicitly normalized networks such as those with [Batch Norm](#batch-norm) hidden layers.

$$
\text{SELU}(x) =
\begin{cases}
\lambda x & \text{if } x > 0 \\
\lambda \alpha (e^x - 1) & \text{if } x \leq 0
\end{cases}
$$

Where the constants are typically:
- λ≈1.0507
- α≈1.67326

## Parameters
This actvation function does not have any parameters.

## Plots
<img src="../../images/activation-functions/selu.png" alt="SELU Function" width="500" height="auto">

<img src="../../images/activation-functions/selu-derivative.png" alt="SELU Derivative" width="500" height="auto">

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SELU\SELU;

$activationFunction = new SELU();
```

## References
[1]: G. Klambauer et al. (2017). Self-Normalizing Neural Networks.
