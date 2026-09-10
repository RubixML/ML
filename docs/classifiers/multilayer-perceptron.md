<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Classifiers/MultilayerPerceptron.php">[source]</a></span>

# Multilayer Perceptron

A multiclass feed-forward neural network classifier with user-defined hidden layers. The Multilayer Perceptron is a deep learning model capable of forming higher-order feature representations through layers of computation. In addition, the MLP features progress monitoring which stops training when it can no longer improve the validation score. It also utilizes network snapshotting to make sure that it always has the best model parameters even if progress began to decline during training.

!!! note
    If there are not enough training samples to build an internal validation set with the user-specified holdout ratio then progress monitoring will be disabled.

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
| 7 | minChange | 1e-4 | float | The minimum change in the training loss necessary to continue training. |
| 8 | evalInterval | 3 | int | The number of epochs to train before evaluating the model using the holdout set. |
| 9 | window | 5 | int | The number of epochs without improvement in the validation score to wait before considering an early stop. |
| 10 | holdOut | 0.1 | float | The proportion of training samples to use for internal validation. Set to 0 to disable. |
| 11 | costFn | MulticlassCrossEntropy | ClassificationLoss | The function that computes the loss associated with an erroneous activation during training. |
| 12 | metric | FBeta | Metric | The validation metric used to score the generalization performance of the model during training. |

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
    holdOut: 0.1,
    costFn: new MulticlassCrossEntropy(),
    metric: new MCC()
);
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

Returns the underlying neural network instance or `null` if untrained:

```php
public network() : Network|null
```

Clean up any leftover state after training. Only do this if you plan to use the model for inference.

```php
public cleanup() : void
```

Export a Graphviz "dot" encoding of the neural network architecture.

```php
public exportGraphviz() : Encoding
```

```php
use Rubix\ML\Helpers\Graphviz;
use Rubix\ML\Persisters\Filesystem;

$dot = $estimator->exportGraphviz();

Graphviz::dotToImage($dot)->saveTo(new Filesystem('network.png'));
```

![Neural Network Graph](https://github.com/RubixML/ML/blob/master/docs/images/neural-network-graph.png?raw=true)

Set the path of the temporary snapshot file used to store network parameters during training.

```php
public setSnapshotPath(?string $path) : void
```

## References

[^1]: G. E. Hinton. (1989). Connectionist learning procedures.
[^2]: L. Prechelt. (1997). Early Stopping - but when?
[^3]: R. Pascanu, et al. (2013). On the difficulty of training recurrent neural networks.
