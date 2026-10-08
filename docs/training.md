# Training

Most estimators have the ability to be trained with data. Estimators that require training are called [Learners](learner.md) and implement the `train()` method among others. Training is the process of feeding data to the learner so that it can build an internal representation (or *model*). Supervised learners require a [Labeled](datasets/labeled.md) training set. Unsupervised learners can be trained with either a Labeled or [Unlabeled](datasets/unlabeled.md) dataset but only the information contained within the features are used to build the model. Depending on the size of your dataset and choice of learning algorithm, training can a long time or just a few seconds. We recommend assessing your time (compute) and memory requirements before training large models.

To begin training a learner, pass a training Dataset object to the `train()` method on the learner instance like in the example below.

```php
$estimator->train($dataset);
```

We can verify that a learner has been trained by calling the `trained()` method which returns true if the estimator is ready to make predictions.

```php
var_dump($estimator->trained());
```

```sh
bool(true)
```

## Batch vs Online Learning

Batch learning is when a learner is trained in full using only one dataset in a single session. Calling the `train()` method on a learner instance is an example of batch learning. In contrast, *online* learning occurs when a learner is trained over multiple sessions with multiple datasets as small as a single sample each. Learners that are capable of being partially trained like this implement the [Online](online.md) interface which includes the `partial()` method for training with a dataset in an online scheme. Subsequent calls to the `partial()` method will continue training where the learner left off. Online learning is especially useful for when you have a dataset that is too large to fit into memory all at once or when your dataset is in the form of a stream.

```php
$estimator->train($dataset1);

$estimator->partial($dataset2);

$estimator->partial($dataset3);
```

!!! note
    After the initial training, the learner will expect subsequent training sets to contain the same number and order of features.

## Monitoring Progress

Since training is often an iterative process, it is useful to obtain feedback as to how the learner is progressing in real-time. For example, you may want to monitor the training loss to make sure that it isn't increasing instead of decreasing with training. Such early feedback saves you time by allowing you to abort training early if things aren't going well. Learners that implement the [Verbose](verbose.md) interface accept a [PSR-3](https://www.php-fig.org/psr/psr-3/) logger instance that can be used to output training information at each time step (or *epoch*). The library comes built-in with the [Screen](loggers/screen.md) logger that does the job for most cases.

```php
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\Loggers\Screen;

$estimator = new LogisticRegression(batchSize: 128, optimizer: new Adam(scheduler: new Constant(0.01)));

$estimator->setLogger(new Screen());

$estimator->train($dataset);
```

```text
[2020-09-04 08:39:04] INFO: Logistic Regression (batch_size: 128, optimizer: Adam (scheduler: Constant (rate: 0.01), momentum decay: 0.1, norm decay: 0.001), alpha: 0.0001, epochs: 1000, min_change: 0.0001, window: 5, cost_fn: Cross Entropy) initialized
[2020-09-04 08:39:04] INFO: Epoch 1 - Cross Entropy: 0.16895133388673
[2020-09-04 08:39:04] INFO: Epoch 2 - Cross Entropy: 0.16559247705179
[2020-09-04 08:39:04] INFO: Epoch 3 - Cross Entropy: 0.16294448401323
[2020-09-04 08:39:04] INFO: Epoch 4 - Cross Entropy: 0.16040135038265
[2020-09-04 08:39:04] INFO: Epoch 5 - Cross Entropy: 0.15786801071483
[2020-09-04 08:39:04] INFO: Epoch 6 - Cross Entropy: 0.1553151426337
[2020-09-04 08:39:04] INFO: Epoch 7 - Cross Entropy: 0.15273253982757
[2020-09-04 08:39:04] INFO: Epoch 8 - Cross Entropy: 0.15011771931339
[2020-09-04 08:39:04] INFO: Epoch 9 - Cross Entropy: 0.14747194148672
[2020-09-04 08:39:04] INFO: Epoch 10 - Cross Entropy: 0.14479847759871
...
[2020-09-04 08:39:04] INFO: Epoch 77 - Cross Entropy: 0.0082096137827592
[2020-09-04 08:39:04] INFO: Epoch 78 - Cross Entropy: 0.0081004235278088
[2020-09-04 08:39:04] INFO: Epoch 79 - Cross Entropy: 0.0079956096838174
[2020-09-04 08:39:04] INFO: Epoch 80 - Cross Entropy: 0.0078948616067878
[2020-09-04 08:39:04] INFO: Epoch 81 - Cross Entropy: 0.0077978960869396
[2020-09-04 08:39:04] INFO: Training complete

```

## Progress Table

In addition to the scalar values returned by the `losses()` and `scores()` accessors, learners and other iterative estimators expose a `progress()` method that returns an iterable table combining every recorded epoch into a single, ordered sequence suitable for inspecting how the model evolved round over round.

This is most useful when the loss and score do not move in tandem — for example, when a training loss continues to drop long after the validation score has begun to plateau or regress. Lining up the epochs side by side makes it possible to identify when the model stopped generalizing and to pick a reasonable point to stop at.

Each entry in the table is an associative array with `Epoch` as one of its keys. The other keys present may vary by estimator; every entry includes the epoch number and the training loss keyed by its name (such as `Exponential Loss` or `Inertia`), and most also include the validation score keyed by the metric name (when a validation dataset was supplied) and the gradient norm (for estimators trained with the neural network subsystem).

```php
use Rubix\ML\Extractors\CSV;

$estimator->train($dataset);

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->progress());
```

The resulting file contains one row per recorded epoch. Values that were not recorded at a given epoch are left blank, such as the validation score on epochs that fall between evaluation intervals.

```csv
Epoch,Exponential Loss,Gradient Norm,F Beta (beta: 1)
1,0.6931,1.2007,
2,0.4034,0.8631,
3,0.2588,0.6102,0.7889
```

Because `progress()` returns an iterator, it can be passed directly to an exporter, plotted, or streamed without loading the entire table into memory.

You can also iterate over it manually to extract or transform individual epochs:

```php
foreach ($estimator->progress() as $record) {
    if (isset($record['F Beta (beta: 1)']) and $record['Exponential Loss'] < 0.3) {
        printf("Loss dropped below 0.3 at epoch %d with a score of %.4F.", $record['Epoch'], $record['F Beta (beta: 1)']);
    }
}
```

!!! note
    A learner must be trained before `progress()` returns any epochs. The `Gradient Norm` and metric keys are only present for epochs where the corresponding value was actually recorded during training.

## Parallel Training

Learners that implement the [Parallel](parallel.md) interface can utilize a parallel processing (multiprocessing) backend for training. Parallel computing can greatly reduce training time on multicore systems at the cost of some overhead to synchronize the data. For small datasets, the overhead may actually cause the runtime to increase. Most parallel learners do not use parallel processing by default, so to enable it you must set a parallel backend using the `setBackend()` method. In the example below, we'll train a [Random Forest](classifiers/random-forest.md) classifier with 500 trees in parallel using the [Amp](backends/amp.md) backend under the hood. By settings the `$workers` argument to 4 we tell the backend to use up to 4 cores at a time to execute the computation.

```php
use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\ClassificationTree;
use Rubix\ML\Backends\Amp;

$estimator = new RandomForest(new ClassificationTree(20), 500);

$estimator->setBackend(new Amp(4));

$estimator->train($dataset);
```

## Feature Importances

Learners that implement the [Ranks Features](ranks-features.md) interface can evaluate the importance of each feature in the training set. Feature importances are defined as the degree to which an individual feature influences the model. Feature importances are useful for feature selection and for helping to explain predictions derived from a model. To output the importance scores, call the `featureImportances()` method of a trained learner that implements the Ranks Features interface.

```php
use Rubix\ML\Regressors\Ridge;

$estimator = new Ridge();

$estimator->train($dataset);

$importances = $estimator->featureImportances();

print_r($importances);
```

```sh
Array
(
    [0] => 0.04757
    [1] => 0.37948
    [2] => 0.53170
    [3] => 0.04123
)
```
