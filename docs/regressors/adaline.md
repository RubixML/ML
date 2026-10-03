<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/Adaline.php">[source]</a></span>

# Adaline

*Adaptive Linear Neuron* is a single layer feed-forward neural network with a continuous linear output neuron suitable for regression tasks. Training is equivalent to solving regularized linear regression with an elastic net penalty online using Mini Batch Gradient Descent. In addition, the learner features progress monitoring which stops training when it can no longer improve the validation score. It also utilizes network snapshotting to make sure that it always has the best model parameters even if progress began to decline during training.

!!! note
    Progress monitoring and early stopping require a validation set. Use `setValidationDataset()` to supply one.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Iterative](../iterative.md), [Online](../online.md), [Ranks Features](../ranks-features.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | batchSize | 128 | int | The number of training samples to process at a time. |
| 2 | optimizer | Adam | Optimizer | The gradient descent optimizer used to update the network parameters. |
| 3 | l1Penalty | 1e-4 | float | The amount of L1 regularization applied to the weights of the output layer. |
| 4 | l2Penalty | 1e-4 | float | The amount of L2 regularization applied to the weights of the output layer. |
| 5 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate over the entire training set before terminating. |
| 6 | minChange | 1e-5 | float | The minimum change in the training loss necessary to continue training. |
| 7 | evalInterval | 1 | int | The number of epochs to train before evaluating the model using the validation set. |
| 8 | window | 10 | int | The number of evaluations without improvement in the validation score to wait before considering an early stop. Set to 0 to disable early stopping. |
| 9 | costFn | LeastSquares | RegressionLoss | The function that computes the loss associated with an erroneous activation during training. |
| 10 | metric | RMSE | Metric | The validation metric used to score the generalization performance of the model during training. |

## Example

```php
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\Regressors\Adaline;

$estimator = new Adaline(
    batchSize: 256,
    optimizer: new Adam(scheduler: new Constant(0.001)),
    l1Penalty: 1e-4,
    l2Penalty: 1e-4,
    epochs: 500,
    minChange: 1e-5,
    evalInterval: 1,
    window: 10,
    costFn: new HuberLoss(alpha: 2.5),
    metric: new RMSE()
);
```

## Additional Methods

Return the loss for each epoch from the last training session.

```php
public losses() : float[]|null
```

Set the dataset used to score the model during training. Once a validation dataset is set, `evalInterval` and `window` determine how often it is scored and when training stops early. Pass `null` to disable progress monitoring and early stopping.

```php
public setValidationDataset(?Labeled $dataset) : void
```

Return the validation score for each epoch from the last training session.

```php
public scores() : float[]|null
```

Returns the underlying neural network instance or `null` if untrained. See [FeedForward](../neural-network/feed-forward.md) for more details.

```php
public network() : FeedForward|null
```

Clean up any leftover state after training. Only do this if you plan to use the model for inference.

```php
public cleanup() : void
```

Set the path of the temporary snapshot file used to store network parameters during training.

```php
public setSnapshotPath(?string $path) : void
```

## References

[^1]: B. Widrow. (1960). An Adaptive "Adaline" Neuron Using Chemical "Memistors".
