# Training
Most estimators in Rubix ML must be trained before they can make predictions. Estimators that require training are called [Learners](learner.md) and implement the `train()` method among others. Training is the process of feeding data to a learner so that it can build an internal representation (or *model*) of the problem. Supervised learners such as classifiers and regressors require a dataset with labels to act as a separate training signal. Unsupervised learners such as clusterers and anomaly detectors can be trained with either a labeled or unlabeled dataset but only really require the raw samples. Every learner has a unique way of approaching the problem space but no matter *how* the learner works under the hood the training API is the same.

To begin training a learner, pass a dataset object to the `train()` method on the learner instance.

**Example**

```php
$estimator->train($dataset);
```

## Batch vs Online Learning
Batch learning is when a learner is trained in full using only one dataset in a single session. Calling the `train()` method on the learner instance is an example of batch learning. In contrast, *online* learning occurs when a learner is trained over multiple sessions with multiple datasets as small as a single sample each. Learners that are capable of being partially trained like this implement the [Online](online.md) interface which includes the `partial()` method for training in an online scheme. Subsequent calls to the `partial()` method will continue training where the learner left off since the last training session.

**Example**

```php
$folds = $dataset->fold(3);

$estimator->partial($folds[0]);

$estimator->partial($folds[1]);

$estimator->partial($folds[2]);
```

## Monitoring Progress
Since training is often an iterative process, it is sometimes useful to obtain real-time feedback as to how the learner is progressing. For example, you may want to monitor the training loss to make sure that it isn't increasing instead of decreasing with training. Such early feedback can indicate model overfitting or improperly tuned hyper-parameters. Learners that implement the [Verbose](verbose.md) interface accept a [PSR-3](https://www.php-fig.org/psr/psr-3/) logger instance that can be used to output training information at each time step (or *epoch*).

Rubix ML comes built-in with a [Screen Logger](other/loggers/screen.md) that does the job for most cases.

**Example**

```php
use Rubix\ML\Other\Loggers\Screen;

$estimator->setLogger(new Screen('example'));
```
