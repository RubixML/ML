<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Schedulers/Cosine.php">[source]</a></span>

# Cosine

A cosine annealing learning-rate schedule that smoothly decays the rate from a starting rate down to an ending rate over a fixed number of steps, then holds the ending rate for the remainder of training.

> **Note:** One *step* is one batch of gradient descent — i.e. one forward and backward pass through the network.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | start | 0.01 | float | The learning rate at the start of training. |
| 2 | end | 0.0001 | float | The learning rate reached at the end of the schedule and held thereafter. |
| 3 | steps | 1000 | int | The number of batches taken to move from the start rate to the end rate. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Cosine;

$scheduler = new Cosine(start: 0.01, end: 0.0001, steps: 500);

$optimizer = new Stochastic(scheduler: $scheduler);
```
