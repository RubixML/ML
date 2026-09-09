<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Adam.php">[source]</a></span>

# Adam

Short for *Adaptive Moment Estimation*, the Adam optimizer pairs a [learning-rate schedule](../schedulers/constant.md) with a momentum and rms-normalized step. In addition to storing an exponentially decaying average of past squared gradients like [RMSprop](rms-prop.md), Adam also keeps an exponentially decaying average of past gradients, similar to [Momentum](momentum.md). Whereas Momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | scheduler | | [Scheduler](../schedulers/constant.md) | The learning-rate schedule that supplies the step size each batch. |
| 2 | momentumDecay | 0.1 | float | The decay rate of the accumulated velocity. |
| 3 | normDecay | 0.001 | float | The decay rate of the rms property. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$optimizer = new Adam(new Constant(0.0001), 0.1, 0.001);
```

## References

[^1]: D. P. Kingma et al. (2014). Adam: A Method for Stochastic Optimization.
