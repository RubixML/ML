<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/GlobalNormClipped.php">[source]</a></span>

# Global Norm Clipped

Global Norm Clipped is a gradient clipping wrapper that bounds the L2 norm of the entire gradient set to a given maximum before delegating the step to the wrapped optimizer. When the global norm of all the gradients exceeds the maximum, every gradient is rescaled by the same factor such that the global norm equals the maximum. This preserves the relative direction of descent across all parameters and is computed once per step before any parameter is updated.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | optimizer | | Optimizer | The wrapped optimizer that performs the parameter updates. |
| 2 | max | 1.0 | float | The maximum L2 norm of the entire gradient set. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\GlobalNormClipped;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$optimizer = new GlobalNormClipped(new Adam(new Constant(0.001)), 1.0);
```