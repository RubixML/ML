# Upgrading from 2.0 to 3.0

Rubix ML 3.0 introduces a number of breaking changes, behavioral differences, and new features. This guide walks you through the changes you'll need to make to upgrade your application, in order of importance. Changes that will cause errors are listed first, followed by changes that may affect the results of your models, and finally the new features you can start using right away.

!!! note
    See the [CHANGELOG on GitHub](https://github.com/RubixML/ML/blob/master/CHANGELOG.md) for a complete list of changes.

## Critical Breaking Changes

These changes will cause exceptions or unexpected behavior in code written for 2.0. You'll need to address each one before your code will run correctly.

### 1. Integers are now a categorical data type

Previously, both integers and floats were considered [continuous](representing-your-data.md) data. In 3.0, only floats are considered continuous — integers are now inferred as [categorical](representing-your-data.md).

```php
use Rubix\ML\DataType;

DataType::detect(1);    // categorical
DataType::detect(1.0);  // continuous
DataType::detect('a');  // categorical
```

This affects you in two important ways:

- Estimators and transformers that require continuous features will now **reject** datasets with integer columns. For example, training a [K Means](clusterers/k-means.md), [Ridge](regressors/ridge.md), or any neural network on a column of `[1, 2, 3]` will throw an `InvalidArgumentException` because the features are no longer continuous.
- A column that mixes integers and floats such as `[1, 2, 3.0]` is no longer homogeneous and will fail **dataset validation** with an `InvalidArgumentException`.

The new [Float Type Converter](transformers/float-type-converter.md) transformer converts integers (and numeric strings) to floats. You can apply it to an existing dataset in place with the `apply()` method, or add it to a [Pipeline](./pipeline.md):

```php
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset->apply(new FloatTypeConverter());
```

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Clusterers\KMeans;

$estimator = new Pipeline([
    new FloatTypeConverter(),
    // ...
], new KMeans(5));
```

Output of certain Transformers such as [One Hot Encoder](transformers/one-hot-encoder.md), [Word Count Vectorizer](transformers/word-count-vectorizer.md), and [Token Hashing Vectorizer](transformers/token-hashing-vectorizer.md) are now interpretted as categorical by default. Use [Float Type Converter](transformers/float-type-converter.md) after the initial transformation to recover the old behavior.

```php
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset->apply(new OneHotEncoder())->apply(new FloatTypeConverter());
```

### 2. Cross Entropy loss was split into Binary and Multiclass

The single `Rubix\ML\NeuralNet\CostFunctions\CrossEntropy` class was removed and replaced with two implementations:

- `Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy` for binary output layers (see [Binary Cross Entropy](neural-network/cost-functions/binary-cross-entropy.md))
- `Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy` for multiclass output layers (see [Multiclass Cross Entropy](neural-network/cost-functions/multiclass-cross-entropy.md))

```php
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;

// before
$mlp = new MultilayerPerceptron(hiddenLayers: [], costFn: new CrossEntropy());

// after
$mlp = new MultilayerPerceptron(hiddenLayers: [], costFn: new MulticlassCrossEntropy());
```

If you did not pass a `CrossEntropy` cost function explicitly, no change is needed — [Logistic Regression](classifiers/logistic-regression.md) defaults to `BinaryCrossEntropy`, while the [MLP](classifiers/multilayer-perceptron.md) and [Softmax Classifier](classifiers/softmax-classifier.md) default to `MulticlassCrossEntropy`.

### 3. Optimizers now take a Scheduler instead of a raw learning rate

The first constructor argument of the neural network optimizers changed from a raw learning rate (`float $rate`) to a [Scheduler](neural-network/schedulers/constant.md) object that supplies the learning rate for each batch. This affects [Adam](neural-network/optimizers/adam.md), [AdaMax](neural-network/optimizers/adamax.md), [RMS Prop](neural-network/optimizers/rms-prop.md), [AdaGrad](neural-network/optimizers/adagrad.md), [Stochastic](neural-network/optimizers/stochastic.md), and [Momentum](neural-network/optimizers/momentum.md). Passing a float instead of a scheduler now throws a `TypeError`.

Three schedulers are available under the `Rubix\ML\NeuralNet\Optimizers\Schedulers` namespace — [Constant](neural-network/schedulers/constant.md), [Step Decay](neural-network/schedulers/step-decay.md), and [Cyclical](neural-network/schedulers/cyclical.md):

```php
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;
use Rubix\ML\NeuralNet\Optimizers\Schedulers\StepDecay;

// before
$optimizer = new Adam(0.001);

// after - a constant rate equivalent to 0.001
$optimizer = new Adam(new Constant(0.001));

// or a decayed rate
$optimizer = new Adam(new StepDecay(0.01));
```

Because learners accept an optimizer as a constructor argument, calls that pass one must also be updated:

```php
use Rubix\ML\NeuralNet\Optimizers\Schedulers\Constant;

// before
$mlp = new MultilayerPerceptron(hiddenLayers: [], optimizer: new Adam(0.01));

// after
$mlp = new MultilayerPerceptron(hiddenLayers: [], optimizer: new Adam(new Constant(0.01)));
```

!!! note
    The schedule advances by one batch automatically during training, so no additional bookkeeping is required on your end.

### 4. The Softmax activation function was removed

The `Rubix\ML\NeuralNet\ActivationFunctions\Softmax` activation function was removed from the library. Softmax is now computed internally by the `Multiclass` output layer, so the default output of multiclass networks is unchanged. However, any code that instantiated `Softmax` directly — for example as an activation in a hidden layer or a custom network — will now throw a fatal error and must use a different activation function:

```php
use Rubix\ML\NeuralNet\ActivationFunctions\Sigmoid;

// before
$layers = [new Activation(new Softmax())];

// after - use another activation such as Sigmoid
$layers = [new Activation(new Sigmoid())];
```

### 5. The L2 Penalty parameter was removed from MLP learners

The `$l2Penalty` constructor parameter was removed from the [Multilayer Perceptron](classifiers/multilayer-perceptron.md) and [MLP Regressor](regressors/mlp-regressor.md). The output layer is no longer regularized directly.

```php
// before
$mlp = new MultilayerPerceptron(hiddenLayers: [], l2Penalty: 1e-4);

// after - regularize via the Dense hidden layers instead
$mlp = new MultilayerPerceptron(hiddenLayers: [new Dense(neurons: 100, l2Penalty: 1e-4)]);
```

!!! note
    `$l2Penalty` is still accepted by [Adaline](regressors/adaline.md), [Logistic Regression](classifiers/logistic-regression.md), and the [Softmax Classifier](classifiers/softmax-classifier.md). Only the MLP learners and the neural net output layers changed.

### 6. TF-IDF dampening was renamed to sublinear

The second constructor parameter of the [TF-IDF Transformer](transformers/tf-idf-transformer.md) was renamed from `$dampening` to `$sublinear`.

```php
use Rubix\ML\Transformers\TfIdfTransformer;

// before
$tfIdf = new TfIdfTransformer(smoothing: 2.0, dampening: true);

// after
$tfIdf = new TfIdfTransformer(smoothing: 2.0, sublinear: true);
```

The parameter occupies the same position (2nd), so code that passes it positionally will continue to work. Named arguments, however, must be updated.

### 7. Exportable Extractors now append by default

[CSV](extractors/csv.md) and NDJSON extractors, as well as the `exportTo()` [Dataset](datasets/api.md) method, no longer overwrite files by default. The `$overwrite` flag defaults to `false`, which means export now *appends* to the existing file contents.

```php
$dataset->exportTo($extractor);           // appends - was overwrite in 2.0
$dataset->exportTo($extractor, true);     // overwrite, as before

$csv->export($iterator, true);            // overwrite, as before
```

!!! warning
    Export runs that previously replaced files will now grow them. Pass `overwrite: true` (or the 2nd positional argument) to preserve 2.0 behavior.

### 8. Spatial trees now constrain their distance kernels

The [Ball Tree](graph/trees/ball-tree.md) and [Vantage Tree](graph/trees/vantage-tree.md) now require a Subadditive distance kernel, and the [K-d Tree](graph/trees/k-d-tree.md) requires a *Monotonic* kernel. Passing any other kernel, including a custom one that doesn't implement the new interfaces, throws an `InvalidArgumentException`.

- Ball Tree / Vantage Tree kernels must implement `Rubix\ML\Kernels\Distance\Subadditive`
- K-d Tree kernels must implement `Rubix\ML\Kernels\Distance\Monotonic`

The default [Euclidean](kernels/distance/euclidean.md) kernel satisfies both, so unless you were passing a custom or an incompatible kernel, no action is required.

### 9. The Backend interface gained a workers() method

The [Backend](backends/amp.md) interface added a `workers()` method that returns the number of concurrent worker processes. If you implemented a custom backend, you must implement it:

```php
use Rubix\ML\Backends\Backend;

class MyBackend implements Backend
{
    public function workers() : int
    {
        // return the number of concurrent workers
    }
}
```

In addition, backend state is no longer serialized — it is now transient per environment. When a model that uses a parallel backend is loaded from disk, a fresh worker pool is constructed on demand rather than restoring the previous pool.

Parallel backends also now default to the number of **physical** CPU cores rather than logical cores.

### 10. The Word Stemmer tokenizer was removed

The `Rubix\ML\Tokenizers\WordStemmer` tokenizer was removed from the library. Use one of the remaining [tokenizers](tokenizers/word.md) such as `Word`, or perform stemming outside of the pipeline with a library of your choice.

```php
use Rubix\ML\Tokenizers\Word;

// before
$tokenizer = new WordStemmer('en');

// after
$tokenizer = new Word();
```

### 11. Updated dependencies

Two dependencies require upgrading on your end if you integrate with them directly:

- **PSR-3 Log v3** — custom [loggers](loggers/screen.md) and `LoggerInterface` implementations must conform to the PSR-3 v3 signatures.
- **Amp v2** — the [Amp Backend](backends/amp.md) now requires `amphp/parallel` ^2.0. If you pin `amphp/parallel` in your project, upgrade it to 2.0.

## Behavioral Changes

These changes won't throw errors, but they can change the output of your models or the shape of your data. Verify that your results are still what you expect.

### 12. Gradient learners now hold out validation data for early stopping

[Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), [Adaline](regressors/adaline.md), and [AdaBoost](classifiers/adaboost.md) now reserve a portion of the training set as a hold-out to drive early stopping, matching the behavior the [MLP](classifiers/multilayer-perceptron.md) learners already had. In 2.0 these learners trained on 100% of the data — now, by default, 10% is held out and the remainder is trained on. Training stops when the validation score does not improve within a window of evaluations.

The relevant constructor parameters (defaults in parentheses) are:

- `$holdOut` — the fraction of samples held out for validation (0.1, must be between 0.0 and 0.5)
- `$evalInterval` — the number of epochs between hold-out evaluations (3)
- `$window` — the number of evaluations without improvement before early stopping (5)

```php
use Rubix\ML\Classifiers\LogisticRegression;

// before - trained on all the data
$lr = new LogisticRegression();

// after - holds out 10% for validation and early stops
$lr = new LogisticRegression(holdOut: 0.1, window: 5, evalInterval: 3);

// to train on all the data without early stopping
$lr = new LogisticRegression(holdOut: 0.0);
```

For [AdaBoost](classifiers/adaboost.md) the change is slightly different — in addition to holding out data, the ensemble is now truncated to the best-performing epoch when early stopping triggers, so later (worse) models are discarded rather than kept.

These parameters are inserted into the constructors after `$minChange`, so calls that pass arguments positionally past that point must be updated (or converted to named arguments). The `$evalInterval` parameter itself is covered in more detail in [item 28](#28-validation-interval-for-hold-out-evaluation).

!!! warning
    Because these learners now see only 90% of your training data by default and may stop early, models fit without explicit configuration may differ from 2.0. Fit with `holdOut: 0.0` or re-tune if results change unexpectedly.

### 13. Token Hashing Vectorizer now defaults to Murmur3

The default hash function of the [Token Hashing Vectorizer](transformers/token-hashing-vectorizer.md) changed from CRC32 to `Murmur3`. Since the hashing function determines which dimensions the tokens map to, the resulting vectors are different from 2.0. Re-fit any pipeline that uses this transformer, or pass `TokenHashingVectorizer::CRC32` explicitly to preserve the previous behavior:

```php
use Rubix\ML\Transformers\TokenHashingVectorizer;

$vectorizer = new TokenHashingVectorizer(100_000, hashFn: TokenHashingVectorizer::CRC32);
```

### 14. V-measure, Completeness, and Homogeneity are now entropy-based

The [V-measure](cross-validation/metrics/v-measure.md), [Completeness](cross-validation/metrics/completeness.md), and [Homogeneity](cross-validation/metrics/homogeneity.md) clustering metrics now use a proper entropy-based formula. Their score ranges are unchanged (0.0 to 1.0), but raw scores are not directly comparable to those produced by 2.0.

### 15. Dataset sort() is now unstable

The [Dataset](datasets/api.md) `sort()` method is no longer stable. Equal elements are not guaranteed to retain their relative order. If your comparisons can produce ties and you rely on the previous order, break ties explicitly in your callback.

### 16. Dataset fold() returns excess samples in the last fold

The `fold()` method of both [Unlabeled](datasets/unlabeled.md) and [Labeled](datasets/labeled.md) datasets now places any samples that do not divide evenly into the *last* fold (previously the behavior was undefined). If `n` samples are folded `k` ways, the first `k - 1` folds contain `floor(n / k)` samples and the last fold receives all of the remainder.

Both `fold()` methods now throw an `InvalidArgumentException` when `k` is greater than the number of samples, preventing empty folds (previously this silently produced `k - 1` empty folds with all samples lumped into the last). The [Labeled](datasets/labeled.md) `stratifiedFold()` method additionally throws when `k` is greater than the number of samples in the *smallest* stratum, since every fold must contain at least one sample of every class.

### 17. Interval Discretizer now outputs integers

The [Interval Discretizer](transformers/interval-discretizer.md) now casts intervals as integers instead of strings. Consumers that expect string output — for example, when feeding a one-hot encoder or writing to CSV — should cast the values to strings. The change aligns with integers now being interpreted as categorical data.

```php
use Rubix\ML\Transformers\IntervalDiscretizer;

$transformer = new IntervalDiscretizer(5); // outputs ints, e.g. 0 .. 4
```

### 18. Persistence changes

A few changes affect [model persistence](model-persistence.md):

- **RBX major-version tracking** — the [RBX serializer](serializers/rbx.md) now tracks the *major* library version rather than the minor version.
- **Revision mismatch warning** — the RBX serializer now emits a warning instead of an exception when a class revision mismatch is detected.
- **Circular-reference compensation** — the class revision hash now compensates for circular references, so the revision check is stable for models with cyclic object graphs.
- **Atomic writes** — the [Filesystem persister](persisters/filesystem.md) now writes files atomically, so writes either fully succeed or leave the previous file intact.
- **SVC class map sidecar** — [SVC](classifiers/svc.md) now saves and restores its class label map via a sidecar file. Re-save any SVC/SVR models trained with 2.x to capture their class maps.

### 19. Boolean Converter now converts truthy and falsy values

The [Boolean Converter](transformers/boolean-converter.md) previously only converted actual PHP booleans. It now converts any truthy or falsy value (such as the strings `'true'`/`'false'`, `'1'`/`'0'`, and the integers `1`/`0`). Review any columns you pass through this transformer for unexpected conversions.

### 20. Polynomial Expander is limited to the 10th degree

The [Polynomial Expander](transformers/polynomial-expander.md) now throws an `InvalidArgumentException` if you request a maximum degree greater than 10.

```php
use Rubix\ML\Transformers\PolynomialExpander;

$transformer = new PolynomialExpander(10); // OK
$transformer = new PolynomialExpander(11); // throws
```

### 21. TSNE window early stopping was removed

The `$window` early-stopping parameter was removed from [t-SNE](transformers/t-sne.md). Adjust any constructor calls that passed it:

```php
// before
$tsne = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6, 5);

// after
$tsne = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6);
```

### 22. Decision Trees now have larger leaf nodes by default

The default maximum leaf node size (`$maxLeafSize`) of the decision-tree learners — [Classification Tree](classifiers/classification-tree.md), [Regression Tree](regressors/regression-tree.md), and the [Extra Tree Classifier](classifiers/extra-tree-classifier.md) and [Extra Tree Regressor](regressors/extra-tree-regressor.md) — increased from 3 to 5. Since leaf nodes may now hold more samples, trees fit with default hyper-parameters may be shallower and their predictions may differ from 2.0.

Pass `maxLeafSize: 3` to recover the previous behavior:

```php
use Rubix\ML\Classifiers\ClassificationTree;

$tree = new ClassificationTree(maxLeafSize: 3);
```

### 23. The He initializer was canonicalized and Xavier 2 is a deprecated alias

The [He initializer](neural-network/initializers/he.md) now draws its weights uniformly from ±√(6/fanIn), matching the canonical *Kaiming* He initialization. The 2.0 implementation used a fan-out-biased formula, so neural networks trained in 3.0 start from different weights and may converge to different results.

In addition, [Xavier 2](neural-network/initializers/xavier-2.md) no longer has its own distribution — it now extends `He` as a deprecated alias. Constructing one emits a deprecation warning. Use [He](neural-network/initializers/he.md) directly in new code:

```php
use Rubix\ML\NeuralNet\Initializers\He;

// before
$initializer = new Xavier2();

// after
$initializer = new He();
```

### 24. Multiclass output gradients were corrected

The `Multiclass` output layer now backpropagates through the softmax Jacobian when the cost function is *not* [Multiclass Cross Entropy](neural-network/cost-functions/multiclass-cross-entropy.md) — for example with [Least Squares](neural-network/cost-functions/least-squares.md) or [Huber Loss](neural-network/cost-functions/huber-loss.md). Previously the gradient was treated as a simple `output - expected` difference, which is incorrect for these losses, so MLPs trained with a non-cross-entropy cost function will now train differently (and more correctly).

In addition, the [K-d Tree](graph/trees/k-d-tree.md) had an edge-pruning correction and an optimized traversal, which may slightly change the results of nearest-neighbor searches that use it.

## New Features

The following changes are additive. They require no action to keep existing code working, but you can take advantage of them as part of your upgrade.

### 25. Parallelized nearest neighbors and Isolation Forest

[K Nearest Neighbors](classifiers/k-nearest-neighbors.md), the [KNN Regressor](regressors/knn-regressor.md), and [Isolation Forest](anomaly-detectors/isolation-forest.md) now implement the [Parallel](parallel.md) interface. K-nearest neighbors splits inference across worker processes, and Isolation Forest splits both training and inference — each tree grows and scores independently.

Like all parallel estimators, they use a Backend to process tasks. The default is the [Serial](backends/serial.md) backend, which runs everything in a single process and behaves exactly as before. To actually parallelize, set one of the multiprocessing backends:

```php
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;

$estimator = new KNearestNeighbors(5);

$estimator->setBackend(new Amp());

// or ...

$estimator->setBackend(new Swoole(16));
```

!!! note
    Number of workers now default to the number of *physical* CPU cores rather than logical cores — see the Backend changes in [item 9](#9-the-backend-interface-gained-a-workers-method).

### 26. Disk-based neural network snapshots

The neural network learners — [MLP](classifiers/multilayer-perceptron.md), [MLP Regressor](regressors/mlp-regressor.md), [Adaline](regressors/adaline.md), [Logistic Regression](classifiers/logistic-regression.md), and [Softmax Classifier](classifiers/softmax-classifier.md) — now stream their parameters to a snapshot file on disk during training. This keeps a copy of the best-performing weights available without holding them in memory, and if training diverges into numerical instability the learner restores from the snapshot instead of the last (possibly unstable) epoch.

By default snapshots are written to a temporary file under `sys_get_temp_dir()`. Point the learner at a specific location with `setSnapshotPath()`, or pass `null` to reset to the default:

```php
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\NeuralNet\ActivationFunctions\SiLU;

$mlp = new MultilayerPerceptron(hiddenLayers: [
    new Dense(neurons: 100),
    new Activation(activationFn: new SiLU()),
]);

$mlp->setSnapshotPath('/var/tmp/mlp-snapshot.dat');
```

### 27. Clearable adaptive optimizer state

Optimizers such as [Adam](neural-network/optimizers/adam.md), [RMS Prop](neural-network/optimizers/rms-prop.md), [AdaGrad](neural-network/optimizers/adagrad.md), and [Momentum](neural-network/optimizers/momentum.md) maintain per-parameter state (gradient caches, momentum velocities) that is only needed during training. The neural network learners now expose a `cleanup()` method that discards this residual state by calling `flush()` on the optimizer — useful before reusing an estimator in a long-running process or to free memory after training:

```php
$mlp->train($dataset);

// free residual optimizer state
$mlp->cleanup();
```

### 28. Validation interval for hold-out evaluation

The windowed gradient-based learners — MLP, [MLP Regressor](regressors/mlp-regressor.md), [Adaline](regressors/adaline.md), [Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), [Gradient Boost](regressors/gradient-boost.md), and [AdaBoost](classifiers/adaboost.md) — now accept a `$evalInterval` constructor parameter (default `3`). It controls how often the hold-out set is scored during training, working in tandem with the `window` parameter for early stopping:

```php
$mlp = new MultilayerPerceptron(hiddenLayers: [new Dense(neurons: 100)], epochs: 1000, evalInterval: 5, window: 10);
```

!!! note
    For [Adaline](regressors/adaline.md), [Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), and [AdaBoost](classifiers/adaboost.md) the `$evalInterval`, `$window`, and `$holdOut` parameters are new additions that also change how these learners train — see [item 12](#12-gradient-learners-now-hold-out-validation-data-for-early-stopping).

    In the MLP and MLP Regressor the new parameter is inserted into the constructor after `minChange` and before `window`, and the `l2Penalty` parameter was removed (see [item 5](#5-the-l2-penalty-parameter-was-removed-from-mlp-learners)). Likewise, the `gradientAccumulationSteps` and `maxGradientNorm` parameters (see [item 33](#33-gradient-accumulation-and-clipping-for-mlp-learners)) are inserted after `batchSize` and `optimizer` respectively. Re-check any constructor calls that pass arguments positionally past the optimizer.

### 29. Per-class and per-cluster variance smoothing

[Gaussian Naive Bayes](classifiers/gaussian-naive-bayes.md) and [Gaussian Mixture](clusterers/gaussian-mixture.md) now compute an independent variance epsilon for *each* class (or cluster) instead of a single global epsilon across all of them. This keeps fitting numerically stable even when classes or clusters have very different variance scales. There is no API change — the existing `$smoothing` parameter behaves as before, only the per-class application of it is new.

### 30. One Hot Encoder category exclusion

The [One Hot Encoder](transformers/one-hot-encoder.md) now accepts a list of `$ignoredCategories` to exclude from encoding. Categories in the list are skipped when the encoder is fitted, so they produce no columns. Only string and integer categories can be ignored:

```php
use Rubix\ML\Transformers\OneHotEncoder;

$encoder = new OneHotEncoder(['unknown', -1]); // ignore these categories
```

### 31. Class Purity and Cluster Purity metrics

Two new ground-truth clustering metrics were added — [Class Purity](cross-validation/metrics/class-purity.md) and [Cluster Purity](cross-validation/metrics/cluster-purity.md). They measure the extent to which each class (or cluster) is dominated by a single cluster (or class), returning a score between 0.0 and 1.0 where higher is better. They are complementary to the entropy-based [V-measure](cross-validation/metrics/v-measure.md), [Completeness](cross-validation/metrics/completeness.md), and [Homogeneity](cross-validation/metrics/homogeneity.md) metrics, and are only compatible with clusterers.

### 32. Float Type Converter

The new [Float Type Converter](transformers/float-type-converter.md) transformer converts integer and numeric-string values to their floating point equivalents. It is the drop-in remedy for the integers-as-categorical change in [item 1](#1-integers-are-now-a-categorical-data-type) — apply it to a dataset directly or add it to a Pipeline so that numeric features are always presented to the estimator as continuous:

```php
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset->apply(new FloatTypeConverter());
```

### 33. Gradient accumulation and clipping for MLP learners

The [MLP](classifiers/multilayer-perceptron.md) and [MLP Regressor](regressors/mlp-regressor.md) accept two new constructor parameters:

- `$gradientAccumulationSteps` (default `1`) — the number of mini-batches to accumulate gradients over before applying an update. Higher values simulate a larger batch size without holding a larger batch in memory.
- `$maxGradientNorm` (default `null`) — clips the global L2 norm of the accumulated gradients to this value before updating, which helps prevent exploding gradients during training.

```php
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;

$mlp = new MultilayerPerceptron(
    hiddenLayers: [new Dense(neurons: 100)],
    gradientAccumulationSteps: 4,
    maxGradientNorm: 1.0,
);
```

### 34. Layer freezing for fine-tuning

The neural network learners expose their underlying network via the `network()` method. Before continuing training with `partial()`, you can freeze the first `k` hidden layers so their parameters stay fixed while the remaining layers keep training — useful for fine-tuning a pretrained model on new data:

```php
use Rubix\ML\Classifiers\MultilayerPerceptron;

$mlp->train($dataset);

// freeze the first 2 hidden layers for fine-tuning
$mlp->network()->freezeFirstKLayers(2);

$mlp->partial($newData);

// unfreeze all layers when done
$mlp->network()->unfreeze();
```
