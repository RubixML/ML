<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Clipped.php">[source]</a></span>

# Clipped

Clipped is a gradient clipping wrapper that bounds the magnitude of each element of the gradient to a given maximum before delegating the update to the wrapped optimizer. By capping the absolute value of every gradient component, Clipped prevents exploding gradients and keeps every step within a bounded range.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | optimizer | | Optimizer | The wrapped optimizer that performs the parameter updates. |
| 2 | max | 1.0 | float | The maximum absolute value of any gradient component. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\Clipped;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$optimizer = new Clipped(new Adam(new Constant(0.001)), 1.0);
```