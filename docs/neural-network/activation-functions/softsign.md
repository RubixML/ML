<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/Softsign.php">[source]</a></span>

# Softsign
A smooth sigmoid-shaped function that squashes the input between -1 and 1.

$$
{\displaystyle Softsign = {\frac {x}{1+|x|}}}
$$

## Parameters
This activation function does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\Softsign;

$activationFunction = new Softsign();
```

## References
[^1]: X. Glorot et al. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks.
