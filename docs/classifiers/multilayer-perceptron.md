<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/MultilayerPerceptron.php">[source]</a></span>

# Multilayer Perceptron

A multiclass feed-forward neural network classifier with user-defined hidden layers. The Multilayer Perceptron is a deep learning model capable of forming higher-order feature representations through layers of computation. In addition, the MLP features progress monitoring which stops training when it can no longer improve the validation score. It also utilizes network snapshotting to make sure that it always has the best model parameters even if progress began to decline during training.

!!! note
    Progress monitoring and early stopping require a validation set. Use `setValidationDataset()` to supply one.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Online](../online.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | hiddenLayers | | array | An array composing the user-specified hidden layers of the network in order. |
| 2 | batchSize | 128 | int | The number of training samples to process at a time. |
| 3 | gradientAccumulationSteps | 1 | int | The number of gradient accumulation steps before updating the network parameters. Higher values simulate a larger batch size. |
| 4 | optimizer | Adam | Optimizer | The gradient descent optimizer used to update the network parameters. |
| 5 | maxGradientNorm | null | float | The maximum L2 norm of the gradient set. When exceeded all gradients are rescaled proportionally so that the global norm equals the maximum. |
| 6 | epochs | 1000 | int | The maximum number of training epochs. i.e. the number of times to iterate over the entire training set before terminating. |
| 7 | minChange | 1e-5 | float | The minimum change in the training loss necessary to continue training. |
| 8 | evalInterval | 1 | int | The number of epochs to train before evaluating the model using the validation set. |
| 9 | window | 10 | int | The number of evaluations without improvement in the validation score to wait before considering an early stop. Set to 0 to disable early stopping. |
| 10 | costFn | MulticlassCrossEntropy | ClassificationLoss | The function that computes the loss associated with an erroneous activation during training. |
| 11 | metric | FBeta | Metric | The validation metric used to score the generalization performance of the model during training. |

## Example

```php
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Dropout;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\Layers\PReLU;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;
use Rubix\ML\CrossValidation\Metrics\MCC;

$estimator = new MultilayerPerceptron(
    hiddenLayers: [
        new Dense(neurons: 200),
        new Activation(activationFn: new LeakyReLU()),
        new Dropout(ratio: 0.3),
        new Dense(neurons: 100),
        new Activation(activationFn: new LeakyReLU()),
        new Dropout(ratio: 0.3),
        new Dense(neurons: 50),
        new PReLU(),
    ],
    batchSize: 128,
    optimizer: new Adam(scheduler: new Constant(0.001)),
    maxGradientNorm: null,
    epochs: 1000,
    minChange: 1e-3,
    evalInterval: 10,
    window: 3,
    costFn: new MulticlassCrossEntropy(),
    metric: new MCC()
);
```

## Additional Methods

Return the loss for each epoch from the last training session.

```php
public losses() : float[]|null
```

Return the progress table combining every epoch recorded during the last training session — the loss, the validation score, and the gradient norm when available — into a single ordered sequence.

```php
public progress() : iterable
```

Return the gradient norm for each epoch from the last training session.

```php
public norms() : float[]|null
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
public network() : ?\Rubix\ML\NeuralNet\Network
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

[^1]: G. E. Hinton. (1989). Connectionist learning procedures.
[^2]: L. Prechelt. (1997). Early Stopping - but when?
[^3]: R. Pascanu, et al. (2013). On the difficulty of training recurrent neural networks.
