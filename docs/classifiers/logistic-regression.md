<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/LogisticRegression.php">[source]</a></span>

# Logistic Regression

A linear classifier that uses the logistic (*sigmoid*) function to estimate the probabilities of exactly two class outcomes. The model parameters (weights and bias) are solved using Mini Batch Gradient Descent with pluggable optimizers and cost functions that run on the neural network subsystem. In addition, the learner features progress monitoring which stops training when it can no longer improve the validation score. It also utilizes network snapshotting to make sure that it always has the best model parameters even if progress began to decline during training.

!!! note
    If there are not enough training samples to build an internal validation set with the user-specified holdout ratio then progress monitoring will be disabled.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Iterative](../iterative.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Ranks Features](../ranks-features.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

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
| 7 | evalInterval | 1 | int | The number of epochs to train before evaluating the model using the holdout set. |
| 8 | window | 10 | int | The number of evaluations without improvement in the validation score to wait before considering an early stop. Set to 0 to disable early stopping. |
| 9 | holdOut | 0.1 | float | The proportion of training samples to use for internal validation. Set to 0 to disable. |
| 10 | costFn | BinaryCrossEntropy | ClassificationLoss | The function that computes the loss associated with an erroneous activation during training. |
| 11 | metric | FBeta | Metric | The validation metric used to score the generalization performance of the model during training. |

## Example

```php
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\CrossValidation\Metrics\MCC;

$estimator = new LogisticRegression(
    batchSize: 64,
    optimizer: new Adam(scheduler: new Constant(0.001)),
    l1Penalty: 1e-4,
    l2Penalty: 1e-4,
    epochs: 100,
    minChange: 1e-5,
    evalInterval: 1,
    window: 10,
    holdOut: 0.1,
    costFn: new BinaryCrossEntropy(),
    metric: new MCC()
);
```

## Additional Methods

Return the loss for each epoch from the last training session.

```php
public losses() : float[]|null
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
