<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/RMSProp.php">[source]</a></span>

# RMS Prop

An adaptive gradient technique that divides the current gradient over a rolling window of the magnitudes of recent gradients. Unlike [AdaGrad](adagrad.md), RMS Prop does not suffer from an infinitely decaying step size.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | scheduler | | [Scheduler](../schedulers/constant.md) | The learning-rate schedule that supplies the step size each batch. |
| 2 | decay | 0.1 | float | The decay rate of the rms property. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\RMSProp;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$optimizer = new RMSProp(new Constant(0.01), 0.1);
```

## References

[^1]: T. Tieleman et al. (2012). Lecture 6e rmsprop: Divide the gradient by a running average of its recent magnitude.
