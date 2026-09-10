<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/NeuralNet/Optimizers/Stochastic.php">[source]</a></span>

# Stochastic

An optimizer based on vanilla Stochastic Gradient Descent that takes a step proportional to the rate supplied by its [learning-rate schedule](../schedulers/constant.md).

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | scheduler | | [Scheduler](../schedulers/constant.md) | The learning-rate schedule that supplies the step size each batch. |

## Example

```php
use Rubix\ML\NeuralNet\Optimizers\Stochastic;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$optimizer = new Stochastic(new Constant(0.01));
```
