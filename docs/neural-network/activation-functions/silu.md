<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/SiLU.php">[source]</a></span>

# SiLU
*Sigmoid-weighted Linear Unit* is a smooth rectified activation function that is not monotonically increasing. Instead, a global minimum functions as an implicit regularizer inhibiting the learning of weights of large magnitudes.

## Parameters
This activation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;

$activationFunction = new SiLU();
```

### References
>- S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning.