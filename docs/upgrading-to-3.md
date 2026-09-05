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

**How to migrate** — Cast integer features to floats. The simplest way is to cast your data before instantiating a dataset object:

```php
$samples = array_map(
    fn ($sample) => array_map(fn ($value) => is_int($value) ? (float) $value : $value, $sample),
    $samples,
);

$dataset = new Labeled($samples, $labels);
```

Alternatively, the new [Float Type Converter](transformers/float-type-converter.md) transformer converts integers (and numeric strings) to floats. You can apply it to an existing dataset in place with the `apply()` method, or add it to a Pipeline:

```php
use Rubix\ML\Pipeline;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset = $dataset->apply(new FloatTypeConverter());

// or inside a pipeline
$estimator = new Pipeline(new FloatTypeConverter(), new KMeans(5));
```

!!! warning
    It is a good idea to cast integer features to floats at the point of extraction (i.e. from CSV, JSON, or a database) so that your dataset objects remain constructible. If a column is *intended* to be categorical, no action is needed.

### 2. Cross Entropy loss was split into Binary and Multiclass

The single `Rubix\ML\NeuralNet\CostFunctions\CrossEntropy` class was removed and replaced with two implementations:

- `Rubix\ML\NeuralNet\CostFunctions\BinaryCrossEntropy` for binary output layers (see [Binary Cross Entropy](neural-network/cost-functions/binary-cross-entropy.md))
- `Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy` for multiclass output layers (see [Multiclass Cross Entropy](neural-network/cost-functions/multiclass-cross-entropy.md))

```php
use Rubix\ML\NeuralNet\CostFunctions\MulticlassCrossEntropy;

// before
$mlp = new MultilayerPerceptron([], costFn: new CrossEntropy());

// after
$mlp = new MultilayerPerceptron([], costFn: new MulticlassCrossEntropy());
```

If you did not pass a `CrossEntropy` cost function explicitly, no change is needed — [Logistic Regression](classifiers/logistic-regression.md) defaults to `BinaryCrossEntropy`, while the [MLP](classifiers/multilayer-perceptron.md), [Softmax Classifier](classifiers/softmax-classifier.md), and [MLP Regressor](regressors/mlp-regressor.md) default to `MulticlassCrossEntropy`.

### 3. The L2 Penalty parameter was removed from MLP learners

The `$l2Penalty` constructor parameter was removed from the [Multilayer Perceptron](classifiers/multilayer-perceptron.md) and [MLP Regressor](regressors/mlp-regressor.md). The output layer is no longer regularized directly.

```php
// before
$mlp = new MultilayerPerceptron([], l2Penalty: 1e-4);

// after - regularize via the Dense hidden layers instead
$mlp = new MultilayerPerceptron([new Dense(100, 1e-4)]);
```

!!! note
    `$l2Penalty` is still accepted by [Adaline](regressors/adaline.md), [Logistic Regression](classifiers/logistic-regression.md), and the [Softmax Classifier](classifiers/softmax-classifier.md). Only the MLP learners and the neural net output layers changed.

### 4. TF-IDF dampening was renamed to sublinear

The second constructor parameter of the [TF-IDF Transformer](transformers/tf-idf-transformer.md) was renamed from `$dampening` to `$sublinear`.

```php
use Rubix\ML\Transformers\TfIdfTransformer;

// before
$tfIdf = new TfIdfTransformer(smoothing: 2.0, dampening: true);

// after
$tfIdf = new TfIdfTransformer(smoothing: 2.0, sublinear: true);
```

The parameter occupies the same position (2nd), so code that passes it positionally will continue to work. Named arguments, however, must be updated.

### 5. Exportable Extractors now append by default

[CSV](extractors/csv.md) and NDJSON extractors, as well as the `exportTo()` [Dataset](datasets/api.md) method, no longer overwrite files by default. The `$overwrite` flag defaults to `false`, which means export now *appends* to the existing file contents.

```php
$dataset->exportTo($extractor);           // appends - was overwrite in 2.0
$dataset->exportTo($extractor, true);     // overwrite, as before

$csv->export($iterator, true);            // overwrite, as before
```

!!! warning
    Export runs that previously replaced files will now grow them. Pass `overwrite: true` (or the 2nd positional argument) to preserve 2.0 behavior.

### 6. Spatial trees now constrain their distance kernels

The [Ball Tree](graph/trees/ball-tree.md) and [Vantage Tree](graph/trees/vantage-tree.md) now require a Subadditive distance kernel, and the [K-d Tree](graph/trees/k-d-tree.md) requires a *Monotonic* kernel. Passing any other kernel, including a custom one, throws an `InvalidArgumentException`.

- Ball Tree / Vantage Tree kernels must implement `Rubix\ML\Kernels\Distance\Subadditive`
- K-d Tree kernels must implement `Rubix\ML\Kernels\Distance\Monotonic`

The default [Euclidean](kernels/distance/euclidean.md) kernel satisfies both, so unless you were passing a custom or an incompatible kernel, no action is required.

### 7. The Backend interface gained a workers() method

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

### 8. The Word Stemmer tokenizer was removed

The `Rubix\ML\Tokenizers\WordStemmer` tokenizer was removed from the library. Use one of the remaining [tokenizers](tokenizers/word.md) such as `Word`, or perform stemming outside of the pipeline with a library of your choice.

```php
use Rubix\ML\Tokenizers\Word;

// before
$tokenizer = new WordStemmer('en');

// after
$tokenizer = new Word();
```

### 9. Updated dependencies

Two dependencies require upgrading on your end if you integrate with them directly:

- **PSR-3 Log v3** — custom [loggers](loggers/screen.md) and `LoggerInterface` implementations must conform to the PSR-3 v3 signatures.
- **Amp v2** — the [Amp Backend](backends/amp.md) now requires `amphp/parallel` ^2.0. If you pin `amphp/parallel` in your project, upgrade it to 2.0.

## Behavioral Changes

These changes won't throw errors, but they can change the output of your models or the shape of your data. Verify that your results are still what you expect.

### 10. Token Hashing Vectorizer now defaults to Murmur3

The default hash function of the [Token Hashing Vectorizer](transformers/token-hashing-vectorizer.md) changed from CRC32 to `Murmur3`. Since the hashing function determines which dimensions the tokens map to, the resulting vectors are different from 2.0. Re-fit any pipeline that uses this transformer, or pass `TokenHashingVectorizer::CRC32` explicitly to preserve the previous behavior:

```php
use Rubix\ML\Transformers\TokenHashingVectorizer;

$vectorizer = new TokenHashingVectorizer(100_000, hashFn: TokenHashingVectorizer::CRC32);
```

### 11. V-measure, Completeness, and Homogeneity are now entropy-based

The [V-measure](cross-validation/metrics/v-measure.md), [Completeness](cross-validation/metrics/completeness.md), and [Homogeneity](cross-validation/metrics/homogeneity.md) clustering metrics now use a proper entropy-based formula. Their score ranges are unchanged (0.0 to 1.0), but raw scores are not directly comparable to those produced by 2.0.

### 12. Dataset sort() is now unstable

The [Dataset](datasets/api.md) `sort()` method is no longer stable. Equal elements are not guaranteed to retain their relative order. If your comparisons can produce ties and you rely on the previous order, break ties explicitly in your callback.

### 13. Dataset fold() returns excess samples in the last fold

The `fold()` method of both [Unlabeled](datasets/unlabeled.md) and [Labeled](datasets/labeled.md) datasets now places any samples that do not divide evenly into the *last* fold (previously the behavior was undefined). If `n` samples are folded `k` ways, the first `k - 1` folds contain `floor(n / k)` samples and the last fold receives all of the remainder.

### 14. Interval Discretizer now outputs integers

The [Interval Discretizer](transformers/interval-discretizer.md) now encodes intervals as integer categories instead of numeric strings. Consumers that expect string output — for example, when feeding a one-hot encoder or writing to CSV — should cast the values to strings. The change aligns with integers now being interpreted as categorical data.

```php
use Rubix\ML\Transformers\IntervalDiscretizer;

$transformer = new IntervalDiscretizer(5); // outputs ints, e.g. 0 .. 4
```

### 15. Persistence changes

A few changes affect [model persistence](model-persistence.md):

- **RBX major-version tracking** — the [RBX serializer](serializers/rbx.md) now tracks the *major* library version rather than the minor version.
- **Revision mismatch warning** — the RBX serializer now emits a warning (instead of failing or proceeding silently) when a class revision mismatch is detected.
- **Atomic writes** — the [Filesystem persister](persisters/filesystem.md) now writes files atomically, so writes either fully succeed or leave the previous file intact.
- **SVC class map sidecar** — [SVC](classifiers/svc.md) now saves and restores its class label map via a sidecar file. Re-save any SVC/SVR models trained with 2.x to capture their class maps.

### 16. Boolean Converter now converts truthy and falsy values

The [Boolean Converter](transformers/boolean-converter.md) previously only converted actual PHP booleans. It now converts any truthy or falsy value (such as the strings `'true'`/`'false'`, `'1'`/`'0'`, and the integers `1`/`0`). Review any columns you pass through this transformer for unexpected conversions.

### 17. Polynomial Expander is limited to the 10th degree

The [Polynomial Expander](transformers/polynomial-expander.md) now throws an `InvalidArgumentException` if you request a maximum degree greater than 10.

```php
use Rubix\ML\Transformers\PolynomialExpander;

$transformer = new PolynomialExpander(10); // OK
$transformer = new PolynomialExpander(11); // throws
```

### 18. TSNE window early stopping was removed

The `$window` early-stopping parameter was removed from [t-SNE](transformers/t-sne.md). Adjust any constructor calls that passed it:

```php
// before
$tsne = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6, 5);

// after
$tsne = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6);
```

## New Features

The following changes are additive and require no action, but are worth knowing about:

- **Parallelized inference and training** — [K Nearest Neighbors](classifiers/k-nearest-neighbors.md), [KNN Regressor](regressors/knn-regressor.md), and [Isolation Forest](anomaly-detectors/isolation-forest.md) now parallelize their computations.
- **Disk-based neural network snapshots** — MLP-style learners (MLP, MLP Regressor, Adaline, Logistic Regression, Softmax) stream parameter snapshots to disk and recover from numerical instability. Set a snapshot path with `setSnapshotPath()`.
- **Clearable optimizer state** — adaptive optimizers such as [Adam](neural-network/optimizers/adam.md) implement `reset()`, which is invoked by the new `cleanup()` method on neural network learners to remove residual state.
- **Validation interval** — windowed gradient-based learners such as the MLP, [Gradient Boost](regressors/gradient-boost.md), [AdaBoost](classifiers/adaboost.md), Logistic Regression, and Softmax now take a `$evalInterval` parameter controlling how often the hold-out set is evaluated during training.
- **Per-class and per-cluster smoothing** — [Gaussian Naive Bayes](classifiers/gaussian-naive-bayes.md) computes a per-class epsilon and [Gaussian Mixture](clusterers/gaussian-mixture.md) a per-cluster epsilon during fitting.
- **One-hot category exclusion** — the [One Hot Encoder](transformers/one-hot-encoder.md) accepts an `$ignoredCategories` array to exclude certain categories from encoding.
- **New clustering metrics** — [Class Purity](cross-validation/metrics/class-purity.md) and [Cluster Purity](cross-validation/metrics/cluster-purity.md) metrics were added.
- **New transformer capabilities** — the [Regex Filter](transformers/regex-filter.md) gained an Emoji preset, and the new [Float Type Converter](transformers/float-type-converter.md) converts numeric strings and integers to floats.
- **Atomic and append-friendly exports** — see the [Exportable Extractors](#5-exportable-extractors-now-append-by-default) and [Persistence](#15-persistence-changes) sections above.