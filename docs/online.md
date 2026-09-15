# Online

Learners that implement the Online interface can be trained in batches. Learners of this type are great for when you either have a continuous stream of data or a dataset that is too large to fit into memory. In addition, partial training allows the model to evolve over time.

## Partially Train

To partially train an Online learner pass it a training set to its `partial()` method:

```php
public partial(Dataset $dataset) : void
```

```php
$folds = $dataset->fold(3);

$estimator->train($folds[0]);

$estimator->partial($folds[1]);

$estimator->partial($folds[2]);
```

!!! note
    Learner will continue to train as long as you are using the `partial()` method, however, calling `train()` on a trained or partially trained learner will reset it back to baseline first.

## Streaming Training

For datasets that are too large to fit into memory all at once, you can stream the records from disk using an [Extractor](extracting-data.md) and train in batches with the `chunked()` dataset factory method. In the example below, we stream the records of a large NDJSON file from disk and partially train an [MLP Regressor](regressors/mlp-regressor.md) that is wrapped in a [Pipeline](pipeline.md) with an elastic transformer to update its fitting as the batches are processed.

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

$estimator = new Pipeline([
    new MinMaxNormalizer(),
], new MLPRegressor([
    new Dense(128),
    new Activation(new SiLU()),
    new Dense(64),
    new Activation(new SiLU()),
    new Dense(1),
], optimizer: new Adam(new Constant(0.001)));

$extractor = new NDJSON('too-large.jsonl');

foreach (Labeled::chunked($extractor, 1024) as $batch) {
    $estimator->partial($batch);
}
```

Because `partial()` warms up a learner on the first call, there is no need to call `train()` beforehand. Every batch must be of the same shape and feature order, and since each batch is validated on construction, the first batch will fail fast if a record is malformed.

!!! note
    Streaming training does not shuffle the dataset, so order may effect training.
