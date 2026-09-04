<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/Adaline.php">[source]</a></span>

# Adaline

*Adaptive Linear Neuron* is a single layer feed-forward neural network with a continuous linear output neuron suitable for regression tasks. Training is equivalent to solving L2 regularized linear regression ([Ridge](ridge.md)) online using Mini Batch Gradient Descent. In addition, the learner features progress monitoring which stops training when it can no longer improve the validation score. It also utilizes network snapshotting to make sure that it always has the best model parameters even if progress began to decline during training.

!!! note
    If there are not enough training samples to build an internal validation set with the user-specified holdout ratio then progress monitoring will be disabled.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Ranks Features](../ranks-features.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | batchSize | 128 | int | The number of training samples to process at a time. |
| 2 | optimizer | Adam | Optimizer | The gradient descent optimizer used to update the network parameters. |
| 3 | l2Penalty | 1e-4 | float | The amount of L2 regularization applied to the weights of the output layer. |
| 4 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate over the entire training set before terminating. |
| 5 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 6 | evalInterval | 3 | int | The number of epochs to train before evaluating the model using the holdout set. |
| 7 | window | 5 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 8 | holdOut | 0.1 | float | The proportion of training samples to use for internal validation. Set to 0 to disable. |
| 9 | costFn | LeastSquares | RegressionLoss | The function that computes the loss associated with an erroneous activation during training. |
| 10 | metric | RMSE | Metric | The validation metric used to score the generalization performance of the model during training. |

## Example

```php
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\Regressors\Adaline;

$estimator = new Adaline(256, new Adam(0.001), 1e-4, 500, 1e-6, 3, 5, 0.1, new HuberLoss(2.5), new RMSE());
```

## Additional Methods

Clean up any leftover state after training. Only do this if you plan to use the model for inference.

```php
public cleanup() : void
```

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

Return the validation score for each epoch from the last training session:

```php
public scores() : float[]|null
```

Return the underlying neural network instance or `null` if untrained:

```php
public network() : Network|null
```

Set the path of the temporary snapshot file used to store network parameters during training:

```php
public setSnapshotPath(?string $path) : void
```

## References

[^1]: B. Widrow. (1960). An Adaptive "Adaline" Neuron Using Chemical "Memistors".
