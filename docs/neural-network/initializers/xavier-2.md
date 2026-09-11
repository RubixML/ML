<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Initializers/Xavier2.php">[source]</a></span>

# Xavier 2

The Xavier 2 initializer is a backward-compatible alias of He. Like He, it draws from a uniform distribution with limits of +/- sqrt(6 / fanIn). It is kept to preserve the name for existing configurations.

!!! note
    Xavier 2 is deprecated, use He instead.

## Parameters

This initializer does not have any parameters.

## Example

```php
use Rubix\ML\NeuralNet\Initializers\Xavier2;

$initializer = new Xavier2();
```

## References

[^1]: X. Glorot et al. (2010). Understanding the Difficulty of Training Deep Feedforward Neural Networks.
