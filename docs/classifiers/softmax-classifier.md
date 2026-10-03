<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/SoftmaxClassifier.php">[source]</a></span>

# Softmax Classifier

A multiclass generalization of [Logistic Regression](logistic-regression.md) using a single layer neural network with a Softmax output layer. In addition, the learner features progress monitoring which stops training when it can no longer improve the validation score. It also utilizes network snapshotting to make sure that it always has the best model parameters even if progress began to decline during training.

!!! note
    Progress monitoring and early stopping require a validation set. Use `setValidationDataset()` to supply one.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Iterative](../iterative.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

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
| 9 | costFn | MulticlassCrossEntropy | ClassificationLoss | The function that computes the loss associated with an erroneous activation during training. |
| 10 | metric | FBeta | Metric | The validation metric used to score the generalization performance of the model during training. |

## Example

```php
use Rubix\ML\Classifiers\SoftmaxClassifier;
use Rubix\ML\NeuralNet\Optimizers\Momentum;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use Rubix\ML\CrossValidation\Metrics\FBeta;

$estimator = new SoftmaxClassifier(
    batchSize: 256,
    optimizer: new Momentum(scheduler: new Constant(0.001)),
    l1Penalty: 1e-4,
    l2Penalty: 1e-4,
    epochs: 300,
    minChange: 1e-5,
    evalInterval: 1,
    window: 10,
    costFn: new MulticlassCrossEntropy(),
    metric: new FBeta()
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

Set the path of the temporary snapshot file used to store network parameters during training.

```php
public setSnapshotPath(?string $path) : void
```

Clean up any leftover state after training. Only do this if you plan to use the model for inference.

```php
public cleanup() : void
```
