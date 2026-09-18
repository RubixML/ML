# Iterative

Learners, estimators, and transformers that implement the `Iterative` interface record their progress over training or transformation and expose it on a per-epoch basis. In addition to the scalar values returned by the `losses()` and `scores()` accessors, the `progress()` method returns an iterable table that combines every recorded epoch into a single, ordered sequence suitable for inspecting how the model evolved round over round.

This is most useful when the loss and score do not move in tandem — for example, when a training loss continues to drop long after the validation score has begun to plateau or regress. Lining up the epochs side by side makes it possible to identify when the model stopped generalizing and to pick a reasonable point to stop at.

## Progress

Return an iterable progress table from the last training session:

```php
public progress() : iterable
```

Each entry in the table is an associative array with `Epoch` as one of its keys. The other keys present may vary by estimator; every entry includes the epoch number and the training loss keyed by its name (such as `Exponential Loss` or `Inertia`), and most also include the validation score keyed by the metric name (when a holdout set was used) and the gradient norm (for estimators trained with the neural network subsystem).

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
