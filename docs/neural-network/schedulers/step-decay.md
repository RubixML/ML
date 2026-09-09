<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Schedulers/StepDecay.php">[source]</a></span>

# Step Decay

A learning-rate schedule that reduces the rate by a factor whenever it reaches a new *floor*. The number of steps needed to reach a new floor is defined by the *steps* hyper-parameter.

> **Note:** One *step* is one batch of gradient descent — i.e. one forward and backward pass through the network.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | rate | 0.01 | float | The initial learning rate. |
| 2 | steps | 100 | int | The size of every floor in steps. i.e. the number of batches to take before applying another factor of decay. |
| 3 | decay | 1e-3 | float | The factor to decrease the learning rate by at each *floor*. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\StepDecay;

$scheduler = new StepDecay(0.1, 50, 1e-3);

$optimizer = new Stochastic($scheduler);
```
