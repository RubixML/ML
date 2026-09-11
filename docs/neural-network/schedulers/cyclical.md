<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Schedulers/Cyclical.php">[source]</a></span>

# Cyclical

A learning-rate schedule that cycles the rate between the lower and upper bound over a designated period, while also decaying the upper bound by a factor at each step. Cyclical learning rates have been shown to help escape bad local minima and saddle points of the gradient.

> **Note:** One *step* is one batch of gradient descent — i.e. one forward and backward pass through the network.

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | lower | 0.001 | float | The lower bound on the learning rate. |
| 2 | upper | 0.006 | float | The upper bound on the learning rate. |
| 3 | length | 2000 | int | The number of batches in every half cycle. |
| 4 | decay | 0.99994 | float | The exponential decay factor to decrease the learning rate by every batch. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Cyclical;

$scheduler = new Cyclical(lower: 0.001, upper: 0.005, length: 1000, decay: 0.99994);

$optimizer = new Stochastic(scheduler: $scheduler);
```

## References

[^1]: L. N. Smith. (2017). Cyclical Learning Rates for Training Neural Networks.
