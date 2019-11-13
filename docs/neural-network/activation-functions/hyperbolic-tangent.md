<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/HyperbolicTangent.php">[source]</a></span>

# Hyperbolic Tangent
S-shaped function that squeezes the input value into an output space between -1 and 1. Tanh has the advantage of being zero centered, however is known to saturate with highly positive or negative input values.

### Parameters
This activation Function does not have any parameters.

### Example
```php
use Rubix\ML\NeuralNet\ActivationFunctions\HyperbolicTangent;

$activationFunction = new HyperbolicTangent();
```