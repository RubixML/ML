<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/SiLU.php">[source]</a></span>

# SiLU
Sigmoid Linear Units are smooth and non-monotonic rectified activation functions. Their inputs are weighted by the [Sigmoid](sigmoid.md) activation function acting as a self-gating mechanism.

## Parameters
This activation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;

$activationFunction = new SiLU();
```

### References
[^1]: S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning.
