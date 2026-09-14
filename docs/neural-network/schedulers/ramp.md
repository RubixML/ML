<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Schedulers/Ramp.php">[source]</a></span>

# Ramp

A linear learning-rate schedule that ramps the rate from a starting rate to an ending rate over a fixed number of steps, then holds the ending rate for the remainder of training. It can be used to warm up the rate from a low start to a high target or to cool it down over time.

> **Note:** One *step* is one batch of gradient descent — i.e. one forward and backward pass through the network.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | start | 0.001 | float | The learning rate at the start of training. |
| 2 | end | 0.01 | float | The learning rate reached at the end of the ramp and held thereafter. |
| 3 | steps | 1000 | int | The number of batches taken to move from the start rate to the end rate. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Ramp;

$scheduler = new Ramp(start: 0.001, end: 0.01, steps: 500);

$optimizer = new Stochastic(scheduler: $scheduler);
```
