<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/ActivationFunctions/HardSigmoid/HardSigmoid.php">[source]</a></span>

# Hard Sigmoid
A piecewise linear approximation of the sigmoid function that is computationally more efficient. The Hard Sigmoid function has an output value between 0 and 1, making it useful for binary classification problems.

$$
\text{HardSigmoid}(x) = \max\left(0,\min\left(1, 0.2x + 0.5\right)\right)
$$

## Parameters
This activation function does not have any parameters.

## Plots
<img src="../../images/activation-functions/hard-sigmoid.png" alt="Hard Sigmoid Function" width="500" height="auto">

<img src="../../images/activation-functions/hard-sigmoid-derivative.png" alt="Hard Sigmoid Derivative" width="500" height="auto">

## Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\HardSigmoid\HardSigmoid;

$activationFunction = new HardSigmoid();
```

## References
[1]: https://en.wikipedia.org/wiki/Hard_sigmoid
