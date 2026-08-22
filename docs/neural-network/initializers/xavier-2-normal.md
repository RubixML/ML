<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Initializers/Xavier2Normal.php">[source]</a></span>

# Xavier 2 Normal
The Xavier 2 Normal initializer draws from a truncated normal distribution with mean 0 and standard deviation equal to (2 / (fanIn + fanOut)) ** 0.25. This initializer is best suited for layers that feed into an activation layer that outputs values between -1 and 1 such as [Hyperbolic Tangent](../activation-functions/hyperbolic-tangent.md) and [Softsign](../activation-functions/softsign.md).

## Parameters
This initializer does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\Initializers\Xavier2Normal;

$initializer = new Xavier2Normal();
```

## References
[^1]: X. Glorot et al. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks.
