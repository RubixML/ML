<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/Adaline.php">[source]</a></span>

# Adaline
*Adaptive Linear Neuron* is a single layer feed-forward neural network with a continuous linear output neuron suitable for regression tasks. Training is equivalent to solving L2 regularized linear regression ([Ridge](ridge.md)) online using Mini Batch Gradient Descent.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Ranks Features](../ranks-features.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | batchSize | 128 | int | The number of training samples to process at a time. |
| 2 | optimizer | Adam | Optimizer | The gradient descent optimizer used to update the network parameters. |
| 3 | l2Penalty | 1e-4 | float | The amount of L2 regularization applied to the weights of the output layer. |
| 4 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate over the entire training set before terminating. |
| 5 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 6 | window | 5 | int | The number of epochs without improvement in the training loss to wait before considering an early stop. |
| 7 | costFn | LeastSquares | RegressionLoss | The function that computes the loss associated with an erroneous activation during training. |

## Example
```php
use Rubix\ML\Regressors\Adaline;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;

$estimator = new Adaline(256, new Adam(0.001), 1e-4, 500, 1e-6, 5, new HuberLoss(2.5));
```

## Additional Methods
Return an iterable progress table with the steps from the last training session:
```php
public steps() : iterable
```

```php
use Rubix\ML\Extractors\CSV;

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->steps());
```

Return the loss for each epoch from the last training session:
```php
public losses() : float[]|null
```

Return the underlying neural network instance or `null` if untrained:
```php
public network() : Network|null
```

## References
[^1]: B. Widrow. (1960). An Adaptive "Adaline" Neuron Using Chemical "Memistors".
