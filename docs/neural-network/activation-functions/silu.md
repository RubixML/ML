<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/SiLU.php">[source]</a></span>

# SiLU
*Sigmoid-weighted Linear Unit* (SiLU) a.k.a. *Swish* is a smooth and non-monotonic rectified activation function. The inputs are weighted by the [Sigmoid](sigmoid.md) activation function acting as a self-gating mechanism. In addition, an inherent global minimum functions as an implicit regularizer.

## Parameters
This activation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;

$activationFunction = new SiLU();
```

### References
>- S. Elwing et al. (2017). Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning.
>- P. Ramachandran er al. (2017). Swish: A Self-gated Activation Function.
