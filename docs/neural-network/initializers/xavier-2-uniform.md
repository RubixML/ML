<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Initializers/Xavier2Uniform.php">[source]</a></span>

# Xavier 2 Uniform
The Xavier 2 Uniform initializer draws from a uniform distribution [-limit, limit] where *limit* is equal to (6 / (fanIn + fanOut)) ** 0.25. This initializer is best suited for layers that feed into an activation layer that outputs values between -1 and 1 such as [Hyperbolic Tangent](../activation-functions/hyperbolic-tangent.md) and [Softsign](../activation-functions/softsign.md).

## Parameters
This initializer does not have any parameters.

## Example
```php
use Rubix\ML\NeuralNet\Initializers\Xavier2Uniform;

$initializer = new Xavier2Uniform();
```

## References
[^1]: X. Glorot et al. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks.
