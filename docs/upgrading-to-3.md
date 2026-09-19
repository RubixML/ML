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

DataType::detect(1);    // Categorical
DataType::detect(1.0);  // Continuous
DataType::detect('a');  // Categorical
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

The `$l2Penalty` constructor parameter was removed from the [Multilayer Perceptron](classifiers/multilayer-perceptron.md) and [MLP Regressor](regressors/mlp-regressor.md). The output projection is no longer regularized directly.

```php
// before
$mlp = new MultilayerPerceptron(hiddenLayers: [], l2Penalty: 1e-4);

// after - regularize via the Dense hidden layers instead
$mlp = new MultilayerPerceptron(hiddenLayers: [new Dense(neurons: 100, l2Penalty: 1e-4)]);
```

!!! note
    `$l2Penalty` is still accepted by linear models such as [Adaline](regressors/adaline.md), [Logistic Regression](classifiers/logistic-regression.md), and the [Softmax Classifier](classifiers/softmax-classifier.md).

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

In addition, the [K-d Tree](graph/trees/k-d-tree.md) had an edge-pruning correction and an optimized traversal, which may slightly change the results of nearest-neighbor searches that use it.

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

### 12. K Means and Fuzzy C Means restrict their distance kernels

[K Means](clusterers/k-means.md) and [Fuzzy C Means](clusterers/fuzzy-c-means.md) now only accept a [Euclidean](kernels/distance/euclidean.md) or [Safe Euclidean](kernels/distance/safe-euclidean.md) distance kernel. Any other kernel — [Manhattan](kernels/distance/manhattan.md), [Cosine](kernels/distance/cosine.md), a custom kernel, etc. — throws an `InvalidArgumentException` at construction:

```php
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Kernels\Distance\Cosine;

// before - any compatible distance kernel was accepted
$clusterer = new KMeans(5, kernel: new Cosine());

// after - only Euclidean or Safe Euclidean allowed
$clusterer = new KMeans(5);          // Euclidean, the default
```

!!! note
    The default kernel is Euclidean, so unless you were passing a custom or non-Euclidean kernel, no action is required. The restriction follows from a change in how both algorithms compute their convergence and seeding distances internally.

### 13. The Dense layer gained an L1 penalty parameter

The [Dense](neural-network/hidden-layers/dense.md) hidden layer constructor now takes an `$l1Penalty` parameter inserted between `$neurons` and `$l2Penalty`. Because it occupies a new *positional* slot, any call that passed `$l2Penalty` (or anything after it) positionally will now bind that value to L1 instead:

```php
use Rubix\ML\NeuralNet\Layers\Dense;

// before - 2nd positional argument was the L2 penalty
$layer = new Dense(128, 1e-4);

// after - 2nd argument is now L1; pass L1 first, then L2
$layer = new Dense(128, 0.0, 1e-4);      // L1 = 0, L2 = 1e-4

// named arguments are unaffected
$layer = new Dense(neurons: 128, l2Penalty: 1e-4);
```

Both the L1 and L2 penalties default to `0.0`, so a `Dense` layer with no explicit penalty behaves exactly as before.

### 14. Adaline, Logistic Regression, and Softmax Classifier are now elastic net

The [Adaline](regressors/adaline.md), [Logistic Regression](classifiers/logistic-regression.md), and [Softmax Classifier](classifiers/softmax-classifier.md) learners were upgraded from a L2-only regularizer to an *elastic net* regularizer with a dedicated `$l1Penalty` parameter. The `$l1Penalty` parameter is inserted immediately after `$optimizer` and immediately before `$l2Penalty`, so any call that passes `$l2Penalty` or any later argument **positionally** will now bind its value to L1:

```php
use Rubix\ML\Regressors\Adaline;

// before - 3rd positional argument was the L2 penalty
$regressor = new Adaline(batchSize: 64, l2Penalty: 1e-4);

// after - pass the L1 penalty first, then the L2 penalty
$regressor = new Adaline(batchSize: 64, l1Penalty: 0.0, l2Penalty: 1e-4);
```

Two things change compared to before:

- `$l1Penalty` now defaults to `1e-4` (matching `$l2Penalty`), so models fit without explicit hyper-parameters now apply an L1 as well as an L2 penalty. To recover the previous L2-only behavior, pass `l1Penalty: 0.0`.
- The parameter order shifted, so named arguments are unaffected but positional arguments are realigned (see [item 13](#13-the-dense-layer-gained-an-l1-penalty-parameter) for the analogous effect on `Dense`).

!!! note
    Named-argument callers such as `new Adaline(batchSize: 64, l2Penalty: 1e-4)` keep working, but the model will now also apply the new default L1 penalty of `1e-4`. Set `l1Penalty: 0.0` to restore prior behavior.

### 15. The `steps()` method was renamed to `progress()`

The `steps()` method, which returned an iterable progress table of the epochs recorded during training, was renamed to `progress()`. This affects most iterative learners — [Adaline](regressors/adaline.md), [MLP](classifiers/multilayer-perceptron.md), [MLP Regressor](regressors/mlp-regressor.md), [Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), [Logit Boost](classifiers/logit-boost.md), [AdaBoost](classifiers/adaboost.md), [Gradient Boost](regressors/gradient-boost.md), [K Means](clusterers/k-means.md), [Fuzzy C Means](clusterers/fuzzy-c-means.md), [Gaussian Mixture](clusterers/gaussian-mixture.md), [Mean Shift](clusterers/mean-shift.md), and [t-SNE](transformers/t-sne.md) — any call to `steps()` will now throw an `Error`:

```php
// before
foreach ($estimator->steps() as $epoch) { /* ... */ }

// after
foreach ($estimator->progress() as $epoch) { /* ... */ }
```

The returned table is now also described by the [Iterative](iterative.md) interface, which the `progress()` method implements. See [item 45](#45-iterative-interface-with-progress-method) for the new interface itself.

### 16. DBSCAN is now a Learner, Probabilistic, and Persistable

In 2.0 the [DBSCAN](clusterers/dbscan.md) clusterer was a plain `Estimator`: it ran the density-based clustering algorithm *inside* `predict()`, so you called `predict()` directly on unlabeled data and it both trained and assigned clusters in one step. In 3.0 it is a proper [Learner](learner.md) — you must call `train()` on your data first, then `predict()` to assign new samples — and it also implements [Probabilistic](probabilistic.md) and [Persistable](persistable.md).

Along with the new interfaces, two constructor changes apply to [DBSCAN](clusterers/dbscan.md).

- `$radius` now defaults to `1.0` (was `0.5`). Pass `radius: 0.5` to recover the old default.
- A new `$weighted` boolean (default `false`) was inserted as the 3rd constructor parameter, in front of `$tree`. When `true`, each neighbor's vote in `predict()`/`proba()` is weighted inversely by its distance. Because it occupies a new *positional* slot, any call that passed `$tree` positionally (as the 3rd argument) will now bind its value to `$weighted` instead:

```php
use Rubix\ML\Graph\Trees\BallTree;

// before - 3rd positional argument was the tree
$clusterer = new DBSCAN(0.5, 5, new BallTree());

// after - pass the weighted flag first, then the tree
$clusterer = new DBSCAN(0.5, 5, false, new BallTree());
```

New methods `trained()`, `tree()`, `proba()`, and `probaSample()` were added alongside the existing `params()`, `type()`, and `compatibility()`.

## Behavioral Changes

These changes won't throw errors, but they can change the output of your models or the shape of your data. Verify that your results are still what you expect.

### 17. Gradient learners now hold out validation data for early stopping

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

These parameters are inserted into the constructors after `$minChange`, so calls that pass arguments positionally past that point must be updated (or converted to named arguments). The `$evalInterval` parameter itself is covered in more detail in [item 37](#37-validation-interval-for-hold-out-evaluation).

!!! warning
    Because these learners now see only 90% of your training data by default and may stop early, models fit without explicit configuration may differ from 2.0. Fit with `holdOut: 0.0` or re-tune if results change unexpectedly.

### 18. Token Hashing Vectorizer now defaults to Murmur3

The default hash function of the [Token Hashing Vectorizer](transformers/token-hashing-vectorizer.md) changed from CRC32 to `Murmur3`. Since the hashing function determines which dimensions the tokens map to, the resulting vectors are different from 2.0. Re-fit any pipeline that uses this transformer, or pass `TokenHashingVectorizer::CRC32` explicitly to preserve the previous behavior:

```php
use Rubix\ML\Transformers\TokenHashingVectorizer;

$vectorizer = new TokenHashingVectorizer(100_000, hashFn: TokenHashingVectorizer::CRC32);
```

### 19. V-measure, Completeness, and Homogeneity are now entropy-based

The [V-measure](cross-validation/metrics/v-measure.md), [Completeness](cross-validation/metrics/completeness.md), and [Homogeneity](cross-validation/metrics/homogeneity.md) clustering metrics now use a proper entropy-based formula. Their score ranges are unchanged (0.0 to 1.0), but raw scores are not directly comparable to those produced by 2.0.

### 20. Dataset sort() is now unstable

The [Dataset](datasets/api.md) `sort()` method is no longer stable. Equal elements are not guaranteed to retain their relative order. If your comparisons can produce ties and you rely on the previous order, break ties explicitly in your callback.

### 21. Dataset fold() returns excess samples in the last fold

The `fold()` method of both [Unlabeled](datasets/unlabeled.md) and [Labeled](datasets/labeled.md) datasets now places any samples that do not divide evenly into the *last* fold (previously the behavior was undefined). If `n` samples are folded `k` ways, the first `k - 1` folds contain `floor(n / k)` samples and the last fold receives all of the remainder.

Both `fold()` methods now throw an `InvalidArgumentException` when `k` is greater than the number of samples, preventing empty folds (previously this silently produced `k - 1` empty folds with all samples lumped into the last). The [Labeled](datasets/labeled.md) `stratifiedFold()` method additionally throws when `k` is greater than the number of samples in the *smallest* stratum, since every fold must contain at least one sample of every class.

### 22. Interval Discretizer now outputs integers

The [Interval Discretizer](transformers/interval-discretizer.md) now casts intervals as integers instead of strings. Consumers that expect string output — for example, when feeding a one-hot encoder or writing to CSV — should cast the values to strings. The change aligns with integers now being interpreted as categorical data.

```php
use Rubix\ML\Transformers\IntervalDiscretizer;

$transformer = new IntervalDiscretizer(5); // outputs ints, e.g. 0 .. 4
```

### 23. Persistence changes

A few changes affect [model persistence](model-persistence.md):

- **RBX major-version tracking** — the [RBX serializer](serializers/rbx.md) now tracks the *major* library version rather than the minor version.
- **Revision mismatch warning** — the RBX serializer now emits a warning instead of an exception when a class revision mismatch is detected.
- **Circular-reference compensation** — the class revision hash now compensates for circular references, so the revision check is stable for models with cyclic object graphs.
- **Atomic writes** — the [Filesystem persister](persisters/filesystem.md) now writes files atomically, so writes either fully succeed or leave the previous file intact.
- **SVC class map sidecar** — [SVC](classifiers/svc.md) now saves and restores its class label map via a sidecar file. Re-save any SVC/SVR models trained with 2.x to capture their class maps.

### 24. Boolean Converter now converts truthy and falsy values

The [Boolean Converter](transformers/boolean-converter.md) previously only converted actual PHP booleans. It now converts any truthy or falsy value (such as the strings `'true'`/`'false'`, `'1'`/`'0'`, and the integers `1`/`0`). Review any columns you pass through this transformer for unexpected conversions.

### 25. Polynomial Expander is limited to the 10th degree

The [Polynomial Expander](transformers/polynomial-expander.md) now throws an `InvalidArgumentException` if you request a maximum degree greater than 10.

```php
use Rubix\ML\Transformers\PolynomialExpander;

$transformer = new PolynomialExpander(10); // OK
$transformer = new PolynomialExpander(11); // throws
```

### 26. TSNE window early stopping was removed

The `$window` early-stopping parameter was removed from [t-SNE](transformers/t-sne.md). Adjust any constructor calls that passed it:

```php
// before
$tsne = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6, 5);

// after
$tsne = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6);
```

### 27. Decision Trees now have larger leaf nodes by default

The default maximum leaf node size (`$maxLeafSize`) of the decision-tree learners — [Classification Tree](classifiers/classification-tree.md), [Regression Tree](regressors/regression-tree.md), and the [Extra Tree Classifier](classifiers/extra-tree-classifier.md) and [Extra Tree Regressor](regressors/extra-tree-regressor.md) — increased from 3 to 5. Since leaf nodes may now hold more samples, trees fit with default hyper-parameters may be shallower and their predictions may differ from 2.0.

Pass `maxLeafSize: 3` to recover the previous behavior:

```php
use Rubix\ML\Classifiers\ClassificationTree;

$tree = new ClassificationTree(maxLeafSize: 3);
```

### 28. The He initializer was canonicalized and Xavier 2 is a deprecated alias

The [He initializer](neural-network/initializers/he.md) now draws its weights uniformly from ±√(6/fanIn), matching the canonical *Kaiming* He initialization. The 2.0 implementation used a fan-out-biased formula, so neural networks trained in 3.0 start from different weights and may converge to different results.

In addition, [Xavier 2](neural-network/initializers/xavier-2.md) no longer has its own distribution — it now extends `He` as a deprecated alias. Constructing one emits a deprecation warning. Use [He](neural-network/initializers/he.md) directly in new code:

```php
use Rubix\ML\NeuralNet\Initializers\He;

// before
$initializer = new Xavier2();

// after
$initializer = new He();
```

### 29. Multiclass output gradients were corrected

The `Multiclass` output layer now backpropagates through the softmax Jacobian when the cost function is *not* [Multiclass Cross Entropy](neural-network/cost-functions/multiclass-cross-entropy.md) — for example with [Relative Entropy](neural-network/cost-functions/relative-entropy.md). Previously the gradient was treated as a simple `output - expected` difference, which is incorrect, so MLPs trained with a non-cross-entropy cost function will now train differently (and more correctly).

### 30. Plus Plus and KMC2 seeders always return unique seeds

The [Plus Plus](clusterers/seeders/plus-plus.md) and [KMC2](clusterers/seeders/k-mc2.md) cluster seeders now reject candidate centroids that duplicate an already-selected one, so they produce exactly `k` *distinct* seeds. Previously a re-drawn sample that coincided with an existing centroid was allowed to be added a second time, which could yield fewer than `k` unique centroids and bias cluster initialization.

```php
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Kernels\Distance\Euclidean;

// both seeders are unchanged in their API — the difference is only in behavior
$seeder = new PlusPlus(kernel: new Euclidean());

$seeder->seed($dataset, 10); // guarantees 10 unique centroids in 3.0
```

There is no API change. The only effect is that [K Means](clusterers/k-means.md) and [Fuzzy C Means](clusterers/fuzzy-c-means.md) now always start from `k` distinct initial centroids, which may shift where they converge relative to 2.0.

### 31. NDJSON exporter preserves zero decimals as floats

The [NDJSON](extractors/ndjson.md) exporter now encodes with the `JSON_PRESERVE_ZERO_FRACTION` flag. Floats whose fractional part is zero — for example `5.0` — are now written as `5.0` in the file instead of `5`. This round-trips through the extractor as a float rather than an integer, which matters now that integers are [categorical data](representing-your-data.md) (see [item 1](#1-integers-are-now-a-categorical-data-type)):

```json
{"feature": 5.0, "label": "A"}
```

!!! note
    Re-extract any NDJSON files that were exported with 2.0 and that rely on whole-number floats being read back as integers. The extractor now preserves them as floats.

### 32. Gradient learners' default `minChange` was lowered from 1e-4 to 1e-5

The default `$minChange` — the minimum change in the training loss necessary for training to continue — of the gradient-based learners was lowered from `1e-4` to `1e-5`. This affects [Adaline](regressors/adaline.md), [MLP](classifiers/multilayer-perceptron.md), [MLP Regressor](regressors/mlp-regressor.md), [Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), [Logit Boost](classifiers/logit-boost.md), [AdaBoost](classifiers/adaboost.md), and [Gradient Boost](regressors/gradient-boost.md).

With a smaller convergence threshold, these learners will train for slightly longer before their loss plateaus, so the loss (and, with hold-out early stopping, the model you end up with) may differ marginally from 2.0. There is no API change — the parameter and its position are the same, only the default value moved. Pass `minChange: 1e-4` to recover the 2.0 default. The clusterers' `minChange` parameters ([K Means](clusterers/k-means.md), [Fuzzy C Means](clusterers/fuzzy-c-means.md)) are unaffected and remain at `1e-4`.

### 33. Prior is now the default Strategy of the Missing Data Imputer

The default imputation `Strategy` for *categorical* columns of the [Missing Data Imputer](transformers/missing-data-imputer.md) changed from [K Most Frequent](strategies/k-most-frequent.md) to [Prior](strategies/prior.md). Both make a frequency-based guess, so imputed values are typically the same, but the two strategies compute the guess slightly differently and can diverge on edges (such as a placeholder category that is itself the most frequent). If your imputed categories depended on the exact K Most Frequent behavior, pass the old default explicitly to keep it:

```php
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Strategies\KMostFrequent;
use Rubix\ML\Strategies\Prior;

// 3.0 default - Prior
$imputer = new MissingDataImputer(categorical: new Prior());

// restore the 2.0 default - K Most Frequent
$imputer = new MissingDataImputer(categorical: new KMostFrequent(1));
```

The default `Strategy` for *continuous* columns remains [Mean](strategies/mean.md).

## New Features

The following changes are additive. They require no action to keep existing code working, but you can take advantage of them as part of your upgrade.

### 34. Parallelized nearest neighbors and Isolation Forest

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

### 35. Disk-based neural network snapshots

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

### 36. Clearable adaptive optimizer state

Optimizers such as [Adam](neural-network/optimizers/adam.md), [RMS Prop](neural-network/optimizers/rms-prop.md), [AdaGrad](neural-network/optimizers/adagrad.md), and [Momentum](neural-network/optimizers/momentum.md) maintain per-parameter state (gradient caches, momentum velocities) that is only needed during training. The neural network learners now expose a `cleanup()` method that discards this residual state by calling `flush()` on the optimizer — useful before reusing an estimator in a long-running process or to free memory after training:

```php
$mlp->train($dataset);

// Free residual optimizer state
$mlp->cleanup();
```

### 37. Validation interval for hold-out evaluation

The windowed gradient-based learners — MLP, [MLP Regressor](regressors/mlp-regressor.md), [Adaline](regressors/adaline.md), [Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), [Gradient Boost](regressors/gradient-boost.md), and [AdaBoost](classifiers/adaboost.md) — now accept a `$evalInterval` constructor parameter (default `3`). It controls how often the hold-out set is scored during training, working in tandem with the `window` parameter for early stopping:

```php
$mlp = new MultilayerPerceptron(hiddenLayers: [new Dense(neurons: 100)], epochs: 1000, evalInterval: 5, window: 10);
```

### 38. Per-class and per-cluster variance smoothing

[Gaussian Naive Bayes](classifiers/gaussian-naive-bayes.md) and [Gaussian Mixture](clusterers/gaussian-mixture.md) now compute an independent variance epsilon for *each* class (or cluster) instead of a single global epsilon across all of them. This keeps fitting numerically stable even when classes or clusters have very different variance scales. There is no API change — the existing `$smoothing` parameter behaves as before, only the per-class application of it is new.

### 39. One Hot Encoder category exclusion

The [One Hot Encoder](transformers/one-hot-encoder.md) now accepts a list of `$ignoredCategories` to exclude from encoding. Categories in the list are skipped when the encoder is fitted, so they produce no columns. Only string and integer categories can be ignored:

```php
use Rubix\ML\Transformers\OneHotEncoder;

$encoder = new OneHotEncoder(['unknown', -1]); // ignore these categories
```

### 40. Class Purity and Cluster Purity metrics

Two new ground-truth clustering metrics were added — [Class Purity](cross-validation/metrics/class-purity.md) and [Cluster Purity](cross-validation/metrics/cluster-purity.md). They measure the extent to which each class (or cluster) is dominated by a single cluster (or class), returning a score between 0.0 and 1.0 where higher is better. They are complementary to the entropy-based [V-measure](cross-validation/metrics/v-measure.md), [Completeness](cross-validation/metrics/completeness.md), and [Homogeneity](cross-validation/metrics/homogeneity.md) metrics, and are only compatible with clusterers.

### 41. Float Type Converter

The new [Float Type Converter](transformers/float-type-converter.md) transformer converts integer and numeric-string values to their floating point equivalents. It is the drop-in remedy for the integers-as-categorical change in [item 1](#1-integers-are-now-a-categorical-data-type) — apply it to a dataset directly or add it to a Pipeline so that numeric features are always presented to the estimator as continuous:

```php
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset->apply(new FloatTypeConverter());
```

### 42. Gradient accumulation and clipping for MLP learners

The [MLP](classifiers/multilayer-perceptron.md) and [MLP Regressor](regressors/mlp-regressor.md) accept two new constructor parameters:

- `$gradientAccumulationSteps` (default `1`) — the number of mini-batches to accumulate gradients over before applying an update. Higher values simulate a larger batch size without holding a larger batch in memory.
- `$maxGradientNorm` (default `null`) — clips the global L2 norm of the accumulated gradients to this value before updating, which helps prevent exploding gradients during training.

```php
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\NeuralNet\Layers\Dense;

$mlp = new MultilayerPerceptron(
    hiddenLayers: [new Dense(neurons: 100)],
    batchSize: 32,
    gradientAccumulationSteps: 4,
    maxGradientNorm: 1.0,
);
```

### 43. Layer freezing for fine-tuning

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

### 44. Dataset chunked() factory for online training

A new static `chunked()` factory was added to the [Dataset](datasets/api.md) object API on both the [Labeled](datasets/labeled.md) and [Unlabeled](datasets/unlabeled.md) datasets. It lazily builds an iterable of fixed-size dataset chunks from a larger iterator, so online and [partial-training](online.md) learners can consume samples in bounded batches without loading the whole table into memory at once:

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Classifiers\NaiveBayes;

$extractor = new NDJSON('data.ndjson');
$learner = new NaiveBayes();

foreach (Labeled::chunked($extractor, 256) as $batch) {
    $learner->partial($batch);
}
```

The second argument is the size of each chunk (default `1024`), and the last chunk may contain fewer samples. The third argument, `$verify` (default `true`), controls whether each chunk is validated as it is produced — pass `false` to skip per-chunk validation for maximum throughput.

### 45. Iterative interface with progress() method

A new [Iterative](iterative.md) interface groups the learners, estimators, and transformers that record their progress epoch by epoch during training or transformation. It exposes a `progress()` method that returns an iterable table combining every recorded epoch — the loss, the validation score (when a hold-out set was used), and, for neural network learners, the gradient norm — into a single ordered sequence:

```php
use Rubix\ML\Extractors\CSV;

$estimator->train($dataset);

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->progress());
```

Most iterative learners — including [Adaline](regressors/adaline.md), [MLP](classifiers/multilayer-perceptron.md), [Logistic Regression](classifiers/logistic-regression.md), [K Means](clusterers/k-means.md), [t-SNE](transformers/t-sne.md), and the others listed in [item 15](#15-the-steps-method-was-renamed-to-progress) — now implement the interface. Because `progress()` returns a `Generator`, it can be streamed to an exporter or plotted without loading the whole table into memory. Learners that already exposed this table via `steps()` (renamed to `progress()` in [item 15](#15-the-steps-method-was-renamed-to-progress)) continue to work unchanged.

### 46. Grid Search fromNamedParams() factory method

A new static `fromNamedParams()` factory was added to [Grid Search](grid-search.md). It lets you specify the hyper-parameters by the *name* of the base learner's constructor parameter instead of by position, so the order no longer matters:

```php
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\Kernels\Distance\Manhattan;

$estimator = GridSearch::fromNamedParams(
    KNearestNeighbors::class,
    [
        'k' => [1, 3, 5, 10],
        'kernel' => [new Euclidean(), new Manhattan()],
        'weighted' => [true, false],
    ]
);
```

Omitted hyper-parameters are filled in with their default from the base learner's constructor. Passing a name that is not a constructor parameter throws an `InvalidArgumentException`. This is purely additive — the existing `GridSearch` constructor is unchanged.

### 47. Grid Search results() table

[Grid Search](grid-search.md) now generates a `results()` table: an iterable of every parameter combination tested, each row paired with its validation score, sorted from best to worst. It is convenient for inspecting the entire search space at a glance or exporting it:

```php
use Rubix\ML\Extractors\CSV;

$estimator->train($dataset);

$extractor = new CSV('results.csv', true);

$extractor->export($estimator->results());
```

The existing `best()` and `scores()` methods are unchanged and continue to return the best combination and the raw scores respectively.
