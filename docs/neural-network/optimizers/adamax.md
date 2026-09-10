<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/AdaMax.php">[source]</a></span>

# AdaMax

A version of the [Adam](adam.md) optimizer that replaces the RMS property with the infinity norm of the past gradients, using a [learning-rate schedule](../schedulers/constant.md) to set the step size each batch. As such, AdaMax is generally more suitable for sparse parameter updates and noisy gradients.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | scheduler | | [Scheduler](../schedulers/constant.md) | The learning-rate schedule that supplies the step size each batch. |
| 2 | momentumDecay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | normDecay | 0.001 | float | The decay rate of the infinity norm. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\AdaMax;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$optimizer = new AdaMax(new Constant(0.0001), 0.1, 0.001);
```

## References

[^1]: D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
