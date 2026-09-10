<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/AdaGrad.php">[source]</a></span>

# AdaGrad

Short for *Adaptive Gradient*, the AdaGrad optimizer pairs a [learning-rate schedule](../schedulers/constant.md) with a step size that varies per parameter, speeding up the learning of parameters that do not change often and slowing down the learning of parameters that do enjoy heavy activity. Due to AdaGrad's infinitely decaying step size, training may be slow or fail to converge using a low learning rate.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | scheduler | | [Scheduler](../schedulers/constant.md) | The learning-rate schedule that supplies the step size each batch. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\AdaGrad;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$optimizer = new AdaGrad(new Constant(0.125));
```

## References

[^1]: J. Duchi et al. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization.
