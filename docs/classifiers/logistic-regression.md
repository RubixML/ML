<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/LogisticRegresion.php">[source]</a></span>

# Logistic Regression
A linear classifier that uses the logistic (*sigmoid*) function to estimate the probabilities of exactly two class outcomes. The model parameters (weights and bias) are solved using Mini Batch Gradient Descent with pluggable optimizers and cost functions that run on the neural network subsystem. In addition, the learner features progress monitoring which stops training when it can no longer improve the validation score. It also utilizes network snapshotting to make sure that it always has the best model parameters even if progress began to decline during training.

!!! note
    If there are not enough training samples to build an internal validation set with the user-specified holdout ratio then progress monitoring will be disabled.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Ranks Features](../ranks-features.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

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
| 9 | costFn | BinaryCrossEntropy | ClassificationLoss | The function that computes the loss associated with an erroneous activation during training. |
| 10 | metric | FBeta | Metric | The validation metric used to score the generalization performance of the model during training. |

## Example

```php
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy;
use Rubix\ML\CrossValidation\Metrics\MCC;

$estimator = new LogisticRegression(64, new Adam(0.001), 1e-4, 100, 1e-4, 3, 5, 0.1, new BinaryCrossEntropy(), new MCC());
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
