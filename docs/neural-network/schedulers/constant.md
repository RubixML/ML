<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Schedulers/Constant.php">[source]</a></span>

# Constant

A learning-rate schedule that returns a fixed rate for the entire duration of training. Pair it with any [optimizer](../optimizers/stochastic.md) when the step size should not adapt over the course of training.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | rate | 0.01 | float | The fixed learning rate that controls the global step size. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$scheduler = new Constant(rate: 0.01);

$optimizer = new Stochastic(scheduler: $scheduler);
```
