<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/SELU.php">[source]</a></span>

# SELU
Scaled Exponential Linear Units (SELU) are a self-normalizing activation function based on the [ELU](#elu) activation function. Neuronal activations of SELU networks automatically converge toward zero mean and unit variance, unlike explicitly normalized networks such as those with [Batch Norm](#batch-norm) hidden layers.

$$
{\displaystyle SELU = 1.0507 {\begin{cases}1.67326 (e^{x}-1)&{\text{if }}x<0\\x&{\text{if }}x\geq 0\end{cases}}}
$$

## Parameters
This actvation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SELU;

$activationFunction = new SELU();
```

## References
[^1]: G. Klambauer et al. (2017). Self-Normalizing Neural Networks.
