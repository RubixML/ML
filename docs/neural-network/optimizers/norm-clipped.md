<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/NormClipped.php">[source]</a></span>

# Norm Clipped

Norm Clipped is a gradient clipping wrapper that bounds the L2 norm of each individual gradient to a given maximum before delegating the update to the wrapped optimizer. Unlike [Clipped](clipped.md) which truncates each element independently, Norm Clipped rescales the entire gradient by a single factor when its magnitude exceeds the maximum, preserving the direction of the gradient.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | optimizer | | Optimizer | The wrapped optimizer that performs the parameter updates. |
| 2 | max | 1.0 | float | The maximum L2 norm of an individual gradient. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\NormClipped;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$optimizer = new NormClipped(new Adam(new Constant(0.001)), 1.0);
```