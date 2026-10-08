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

The new [Float Type Converter](transformers/float-type-converter.md) transformer converts integers (and numeric strings) to floats. You can apply it to an existing dataset in place with the `apply()` method, or add it to a [Pipeline](transformers/pipeline.md):

```php
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset->apply(new FloatTypeConverter());
```

```php
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Clusterers\KMeans;

$pipeline = new Pipeline([
    new FloatTypeConverter(),
    // ...
]);

$clusterer = new KMeans(5);

$clusterer->train($dataset->apply($pipeline));
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

Three dependencies require upgrading on your end:

- **PSR-3 Log v3** — custom [loggers](loggers/screen.md) and `LoggerInterface` implementations must conform to the PSR-3 v3 signatures.
- **Amp v2** — the [Amp Backend](backends/amp.md) now requires `amphp/parallel` ^2.0. If you pin `amphp/parallel` in your project, upgrade it to 2.0.
- **Tensor 4.1** — the library now requires `rubix/tensor` ^4.1, and the Tensor extension must be **4.1 or above**. [GELU](neural-network/activation-functions/gelu.md), [Soft Plus](neural-network/activation-functions/soft-plus.md), and [Huber Loss](neural-network/cost-functions/huber-loss.md) check the version at construction and throw a `RuntimeException` when ext-tensor is loaded but older than 4.1.0 — `composer update rubix/tensor` for the pure PHP package, and `pie install rubix/tensor_ext:^4.1` for the extension. See [Installation](installation.md) for the full requirements.

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

The returned table combines every recorded epoch — the loss, the validation score, and, for neural network learners, the gradient norm — into a single ordered sequence. See [item 52](#52-grid-search-results-table) for the new results() table.

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

### 17. Stratification methods were renamed and now require categorical labels

Two [Labeled](datasets/labeled.md) methods were renamed, and the stratification methods they power now reject continuous labels:

| 2.0 | 3.0 |
| --- | --- |
| `stratifyByLabel()` | `stratifyByClassLabels()` |
| `describeByLabel()` | `describeByClassLabels()` |

The renames make it explicit that these methods operate on *class* labels. `stratifyByClassLabels()` is now the sole implementation behind `stratifiedSplit()` and `stratifiedFold()`, and it validates the label type before doing any work:

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::build($samples, ['red', 'green', 'red']);

// before
$strata = $dataset->stratifyByLabel();  // 3.0: Error - undefined method

// after
$strata = $dataset->stratifyByClassLabels();

$report = $dataset->describeByClassLabels();
```

`stratifyByClassLabels()`, `stratifiedSplit()`, and `stratifiedFold()` all throw an `InvalidArgumentException` when the label is continuous. In 2.0 they grouped samples by *exact* label equality, which is meaningless for a float target — every distinct value became its own stratum, or the whole target collapsed into a single stratum when all values repeated. Rather than silently produce a meaningless "stratification", 3.0 refuses the call and directs you to the binned variants described in [item 53](#53-binned-stratification-for-continuous-labels):

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::build($samples, [1.5, 2.75, 3.1, 4.9]);

$dataset->stratifiedSplit(0.8);  // before: silently stratified by exact float equality
                                // 3.0: throws InvalidArgumentException

$dataset->binnedSplit(0.8);       // 3.0 - stratify over equal frequency bins
```

!!! note
    Only datasets with a *continuous* target are affected — in practice, regression datasets. Classifiers, clusterers (whose cluster labels are integers, and therefore [categorical](representing-your-data.md) — see [item 1](#1-integers-are-now-a-categorical-data-type)), and any other learner with categorical labels are unaffected and continue to use `stratifiedSplit()` and `stratifiedFold()` as before. The [Validators](cross-validation.md) dispatch on the label type for you (see [item 54](#54-validators-now-stratify-continuous-labels-by-bin)).

### 18. Pipeline is now a Transformer decorator

The `Pipeline` moved from the root namespace into `Rubix\ML\Transformers` and changed from a meta-estimator that wrapped a base estimator into a plain [Transformer](transformers/api.md). Code that still imports `Rubix\ML\Pipeline` now dies with a fatal `Error: Class "Rubix\ML\Pipeline" not found`, and once that import is fixed, the two parameters that carried the wrapped estimator and the elastic flag no longer exist:

```php
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Classifiers\SoftmaxClassifier;

// before - transformers, a base estimator, and the elastic flag
$estimator = new Pipeline([new OneHotEncoder()], new SoftmaxClassifier(), true);

// after - transformers only
$pipeline = new Pipeline([new OneHotEncoder()]);
```

Mind how that fails: PHP silently ignores surplus arguments, so the call above throws nothing at all — it just drops the wrapped estimator and the elastic flag on the floor. The error surfaces later, from a method the pipeline no longer has: `Error: Call to undefined method Rubix\ML\Transformers\Pipeline::train()`. There is no `$elastic` flag anymore either: a pipeline is [Elastic](transformers/api.md#elastic) whenever at least one transformer in its stack is, and `update()` then refines those fittings while lazily fitting any stateful transformer it has not yet seen.

Everything that made the 2.0 Pipeline a meta-estimator is gone along with the estimator it held:

| 2.0 | 3.0 |
| --- | --- |
| `Estimator`, `Learner`, `Online`, `Probabilistic`, `Scoring`, `Persistable` | `Transformer`, [Stateful](transformers/api.md#stateful), [Elastic](transformers/api.md#elastic), [Persistable](persistable.md) |
| `train()`, `partial()`, `predict()`, `proba()`, `score()`, `base()`, `__call()` | `fit()`, `fitted()`, `update()`, `transform()` |
| `compatibility()` from the base estimator | `compatibility()` from the first transformer in the stack |
| `params()` returns `transformers`, `estimator`, `elastic` | `params()` returns `transformers` |

Since a pipeline no longer holds an estimator, it transforms nothing implicitly. Apply it to the dataset yourself at both training and inference time:

```php
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Classifiers\SoftmaxClassifier;

$pipeline = new Pipeline([new OneHotEncoder()]);

$estimator = new SoftmaxClassifier();

$dataset->apply($pipeline);

$estimator->train($dataset);

$predictions = $estimator->predict($dataset);
```

[apply()](datasets/api.md) transforms the dataset *in place*, fits the pipeline first if it isn't fitted yet, and returns the same object — hence the `clone`, which preserves your untransformed table for the next call. Calling `fit()` yourself is optional, and unlike 2.0 it no longer mutates the dataset you hand it: the pipeline streams a working copy of the samples through the chain and leaves the input alone. Note that fitting only fits — transform separately with `transform()` or `apply()`.

[Online](online.md) learners get the same treatment: nothing is updated for you, so update the pipeline and transform each batch explicitly.

```php
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Extractors\NDJSON;
use Rubix\ML\Regressors\MLPRegressor;

$pipeline = new Pipeline([
    new MinMaxNormalizer(),  // Elastic - refined by every update()
    new OneHotEncoder(),
]);

$regressor = new MLPRegressor($layers);

foreach (Labeled::chunked(new NDJSON('data.jsonl')) as $batch) {
    $pipeline->update($batch);

    $batch->apply($pipeline);

    $regressor->partial($batch);
}
```

Finally, fitted transformer state is no longer persisted along with the estimator — 2.0 saved the whole wrapped pipeline in the same file as the model. A pipeline is a [Persistable](persistable.md) transformer in its own right now, so fit and save it separately, most conveniently with the [Persistent Transformer](transformers/persistent-transformer.md) decorator covered in [item 58](#58-persistenttransformer-decorator).

!!! warning
    A 2.x model whose persisted state embeds a `Rubix\ML\Pipeline` cannot be restored by 3.0 — the class no longer exists. Re-fit your transformers and re-save the model. See [Pipeline](transformers/pipeline.md) for the full reference.

### 19. The Trainable interface was removed

The `Rubix\ML\Trainable` interface was removed from the library. Its two methods — `train()` and `trained()` — were transferred to the [Learner](learner.md) interface, which is now their only home:

```php
// before
use Rubix\ML\Trainable;
use Rubix\ML\Datasets\Dataset;

class MyRegressor implements Trainable
{
    public function train(Dataset $dataset) : void
    {
        // ...
    }

    public function trained() : bool
    {
        // ...
    }
}

// after
use Rubix\ML\Learner;
use Rubix\ML\Datasets\Dataset;

class MyRegressor implements Learner
{
    // the same train() and trained() methods
}
```

Any code that names the interface — a `use Rubix\ML\Trainable;` import, an `implements Trainable` clause, or an `instanceof Trainable` check — now dies with `Error: Interface "Rubix\ML\Trainable" not found`.

The removal has a second, less obvious effect: **`Learner` no longer extends `Estimator`**. In 2.0 the interface was declared `interface Learner extends Trainable, Estimator`, so a `Learner` type hint implied `predict()`, `type()`, `compatibility()`, and `params()`. In 3.0 a bare `Learner` only guarantees `train()` and `trained()`, so anywhere you need both you have to declare the intersection:

```php
use Rubix\ML\Learner;
use Rubix\ML\Estimator;

// before - a Learner was always an Estimator too
function best(Learner $learner) : Estimator
{
    // ...
}

// after - ask for both explicitly
function best(Learner & Estimator $learner) : Estimator
{
    // ...
}
```

The library's own signatures were updated the same way, so anything that implements or overrides them must follow:

- [Validator::test()](cross-validation/api.md) is now declared `test(Learner & Estimator $estimator, Labeled $dataset, Metric $metric) : float`. A custom validator that keeps the old `Learner $estimator` parameter fails with a fatal declaration-compatibility `Error`.
- [PersistentModel](persistent-model.md) now type-hints its base learner as `Learner & Estimator`, so handing the constructor a learner-only object throws a `TypeError`. Its `restore()` method throws an `InvalidArgumentException` when the deserialized base implements only one of the two interfaces.
- [Grid Search](grid-search.md) validates its base class against both interfaces — the constructor error now reads *"Base class must implement the Learner and Estimator Interfaces."*
- [Ranks Features](ranks-features.md) now extends `Learner` instead of `Trainable`, so rankers carry the same two-method contract.

!!! note
    Implementing `Learner` alone is no longer enough to be cross-validated, tuned, or persisted. Declare `implements Learner, Estimator` — as every built-in estimator does — to keep passing the [Validators](cross-validation.md), [Grid Search](grid-search.md), and [PersistentModel](persistent-model.md).

## Behavioral Changes

These changes won't throw errors, but they can change the output of your models or the shape of your data. Verify that your results are still what you expect.

### 20. Gradient learners now support early stopping

[Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), [Adaline](regressors/adaline.md), [AdaBoost](classifiers/adaboost.md), and the other windowed learners now support progress monitoring and early stopping. They always train on the *entire* dataset given to `train()` — no portion of it is carved out. To enable early stopping, supply the validation set yourself with `setValidationDataset()`, which is covered in [item 57](#57-early-stopping-is-opt-in-via-user-supplied-validation-sets). Training stops when the validation score fails to improve within a window of evaluations.

The relevant constructor parameters are:

- `$evalInterval` — the number of epochs between validation evaluations (1, or 3 for the boosting learners)
- `$window` — the number of evaluations without improvement before early stopping (10, or 5 for the boosting learners)

```php
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Datasets\Labeled;

$lr = new LogisticRegression(window: 5, evalInterval: 3);

$lr->setValidationDataset($testing);

$lr->train($training);
```

Both parameters are inert until a validation dataset is set. `$evalInterval` is covered in more detail in [item 42](#42-validation-interval-for-early-stopping-evaluation).

!!! note
    With no validation set, the learner trains on all of the data and validation-score-based monitoring is inactive; loss-based stopping via `minChange` still applies. Remove any `holdOut` argument when migrating, and shift positional arguments that followed it.

### 21. Token Hashing Vectorizer now defaults to Murmur3

The default hash function of the [Token Hashing Vectorizer](transformers/token-hashing-vectorizer.md) changed from CRC32 to `Murmur3`. Since the hashing function determines which dimensions the tokens map to, the resulting vectors are different from 2.0. Re-fit any pipeline that uses this transformer, or pass `TokenHashingVectorizer::CRC32` explicitly to preserve the previous behavior:

```php
use Rubix\ML\Transformers\TokenHashingVectorizer;

$vectorizer = new TokenHashingVectorizer(100_000, hashFn: TokenHashingVectorizer::CRC32);
```

### 22. V-measure, Completeness, and Homogeneity are now entropy-based

The [V-measure](cross-validation/metrics/v-measure.md), [Completeness](cross-validation/metrics/completeness.md), and [Homogeneity](cross-validation/metrics/homogeneity.md) clustering metrics now use a proper entropy-based formula. Their score ranges are unchanged (0.0 to 1.0), but raw scores are not directly comparable to those produced by 2.0.

### 23. Dataset sort() is now unstable

The [Dataset](datasets/api.md) `sort()` method is no longer stable. Equal elements are not guaranteed to retain their relative order. If your comparisons can produce ties and you rely on the previous order, break ties explicitly in your callback.

### 24. Dataset fold() distributes the remainder across all folds

The `fold()` method of both [Unlabeled](datasets/unlabeled.md) and [Labeled](datasets/labeled.md) datasets forms folds that are as equal size as possible. If `n` samples are folded `k` ways, the first `n % k` folds contain `ceil(n / k)` samples and the remaining folds contain `floor(n / k)`. Previously the last fold received the entire remainder, which could leave it holding many times the samples of its siblings.

The [Labeled](datasets/labeled.md) `stratifiedFold()` and `binnedFold()` methods no longer call `fold()` on each stratum independently. Doing so sent the remainder of *every* stratum to the final fold, so a dataset whose strata did not divide evenly produced a single oversized fold — 143 samples across 10 bins folded 10 ways gave sizes of `[10, ..., 10, 53]` instead of `[14, ..., 15]`. Each stratum now awards its remainder to a rotating window of folds whose cursor carries over between strata, which keeps the aggregate fold sizes within a single sample of one another and keeps the proportions of every stratum intact. See [item 53](#53-binned-stratification-for-continuous-labels) for the binned variant.

Both `fold()` methods now throw an `InvalidArgumentException` when `k` is greater than the number of samples, preventing empty folds (previously this silently produced `k - 1` empty folds with all samples lumped into the last). The [Labeled](datasets/labeled.md) `stratifiedFold()` method additionally throws when `k` is greater than the number of samples in the *smallest* stratum, since every fold must contain at least one sample of every class.

### 25. Interval Discretizer now outputs integers

The [Interval Discretizer](transformers/interval-discretizer.md) now casts intervals as integers instead of strings. Consumers that expect string output — for example, when feeding a one-hot encoder or writing to CSV — should cast the values to strings. The change aligns with integers now being interpreted as categorical data.

```php
use Rubix\ML\Transformers\IntervalDiscretizer;

$transformer = new IntervalDiscretizer(5); // outputs ints, e.g. 0 .. 4
```

### 26. Persistence changes

A few changes affect [model persistence](model-persistence.md):

- **RBX major-version tracking** — the [RBX serializer](serializers/rbx.md) now tracks the *major* library version rather than the minor version.
- **Revision mismatch warning** — the RBX serializer now emits a warning instead of an exception when a class revision mismatch is detected.
- **Circular-reference compensation** — the class revision hash now compensates for circular references, so the revision check is stable for models with cyclic object graphs.
- **Atomic writes** — the [Filesystem persister](persisters/filesystem.md) now writes files atomically, so writes either fully succeed or leave the previous file intact.
- **SVC class map sidecar** — [SVC](classifiers/svc.md) now saves and restores its class label map via a sidecar file. Re-save any SVC/SVR models trained with 2.x to capture their class maps.

### 27. Boolean Converter now converts truthy and falsy values

The [Boolean Converter](transformers/boolean-converter.md) previously only converted actual PHP booleans. It now converts any truthy or falsy value (such as the strings `'true'`/`'false'`, `'1'`/`'0'`, and the integers `1`/`0`). Review any columns you pass through this transformer for unexpected conversions.

### 28. Polynomial Expander is limited to the 10th degree

The [Polynomial Expander](transformers/polynomial-expander.md) now throws an `InvalidArgumentException` if you request a maximum degree greater than 10.

```php
use Rubix\ML\Transformers\PolynomialExpander;

$transformer = new PolynomialExpander(10); // OK
$transformer = new PolynomialExpander(11); // throws
```

### 29. TSNE window early stopping was removed

The `$window` early-stopping parameter was removed from [t-SNE](transformers/t-sne.md). Adjust any constructor calls that passed it:

```php
// before
$tsne = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6, 5);

// after
$tsne = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6);
```

### 30. Decision Trees now have larger leaf nodes by default

The default maximum leaf node size (`$maxLeafSize`) of the decision-tree learners — [Classification Tree](classifiers/classification-tree.md), [Regression Tree](regressors/regression-tree.md), and the [Extra Tree Classifier](classifiers/extra-tree-classifier.md) and [Extra Tree Regressor](regressors/extra-tree-regressor.md) — increased from 3 to 5. Since leaf nodes may now hold more samples, trees fit with default hyper-parameters may be shallower and their predictions may differ from 2.0.

Pass `maxLeafSize: 3` to recover the previous behavior:

```php
use Rubix\ML\Classifiers\ClassificationTree;

$tree = new ClassificationTree(maxLeafSize: 3);
```

### 31. The He initializer was canonicalized and Xavier 2 is a deprecated alias

The [He initializer](neural-network/initializers/he.md) now draws its weights uniformly from ±√(6/fanIn), matching the canonical *Kaiming* He initialization. The 2.0 implementation used a fan-out-biased formula, so neural networks trained in 3.0 start from different weights and may converge to different results.

In addition, [Xavier 2](neural-network/initializers/xavier-2.md) no longer has its own distribution — it now extends `He` as a deprecated alias. Constructing one emits a deprecation warning. Use [He](neural-network/initializers/he.md) directly in new code:

```php
use Rubix\ML\NeuralNet\Initializers\He;

// before
$initializer = new Xavier2();

// after
$initializer = new He();
```

### 32. Multiclass output gradients were corrected

The `Multiclass` output layer now backpropagates through the softmax Jacobian when the cost function is *not* [Multiclass Cross Entropy](neural-network/cost-functions/multiclass-cross-entropy.md) — for example with [Relative Entropy](neural-network/cost-functions/relative-entropy.md). Previously the gradient was treated as a simple `output - expected` difference, which is incorrect, so MLPs trained with a non-cross-entropy cost function will now train differently (and more correctly).

### 33. Plus Plus and KMC2 seeders always return unique seeds

The [Plus Plus](clusterers/seeders/plus-plus.md) and [KMC2](clusterers/seeders/k-mc2.md) cluster seeders now reject candidate centroids that duplicate an already-selected one, so they produce exactly `k` *distinct* seeds. Previously a re-drawn sample that coincided with an existing centroid was allowed to be added a second time, which could yield fewer than `k` unique centroids and bias cluster initialization.

```php
use Rubix\ML\Clusterers\Seeders\PlusPlus;
use Rubix\ML\Kernels\Distance\Euclidean;

// both seeders are unchanged in their API — the difference is only in behavior
$seeder = new PlusPlus(kernel: new Euclidean());

$seeder->seed($dataset, 10); // guarantees 10 unique centroids in 3.0
```

There is no API change. The only effect is that [K Means](clusterers/k-means.md) and [Fuzzy C Means](clusterers/fuzzy-c-means.md) now always start from `k` distinct initial centroids, which may shift where they converge relative to 2.0.

### 34. NDJSON exporter preserves zero decimals as floats

The [NDJSON](extractors/ndjson.md) exporter now encodes with the `JSON_PRESERVE_ZERO_FRACTION` flag. Floats whose fractional part is zero — for example `5.0` — are now written as `5.0` in the file instead of `5`. This round-trips through the extractor as a float rather than an integer, which matters now that integers are [categorical data](representing-your-data.md) (see [item 1](#1-integers-are-now-a-categorical-data-type)):

```json
{"feature": 5.0, "label": "A"}
```

!!! note
    Re-extract any NDJSON files that were exported with 2.0 and that rely on whole-number floats being read back as integers. The extractor now preserves them as floats.

### 35. Gradient learners' default `minChange` was lowered from 1e-4 to 1e-5

The default `$minChange` — the minimum change in the training loss necessary for training to continue — of the gradient-based learners was lowered from `1e-4` to `1e-5`. This affects [Adaline](regressors/adaline.md), [MLP](classifiers/multilayer-perceptron.md), [MLP Regressor](regressors/mlp-regressor.md), [Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), [Logit Boost](classifiers/logit-boost.md), [AdaBoost](classifiers/adaboost.md), and [Gradient Boost](regressors/gradient-boost.md).

With a smaller convergence threshold, these learners will train for slightly longer before their loss plateaus, so the loss (and, with hold-out early stopping, the model you end up with) may differ marginally from 2.0. There is no API change — the parameter and its position are the same, only the default value moved. Pass `minChange: 1e-4` to recover the 2.0 default. The clusterers' `minChange` parameters ([K Means](clusterers/k-means.md), [Fuzzy C Means](clusterers/fuzzy-c-means.md)) are unaffected and remain at `1e-4`.

### 36. Prior is now the default Strategy of the Missing Data Imputer

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

### 37. Image Rotator constructor defaults changed

The [Image Rotator](transformers/image-rotator.md) constructor defaults have changed:

- `$offset` is now optional and defaults to `0.0` (it was required in 2.0)
- `$jitter` defaults to `0.2` (it was `0.0` in 2.0)

Because `$jitter` is applied as a fraction of a half-turn (up to ±180°×`$jitter`), calling `new ImageRotator()` now adds random jitter of up to ±36° by default. To preserve 2.0 behavior, pass explicit values:

```php
use Rubix\ML\Transformers\ImageRotator;

// 3.0 default - random jitter about the origin, up to ±36 degrees
$transformer = new ImageRotator();

// 3.0 - fixed rotation, still jittered
$transformer = new ImageRotator(-90.0);

// to recover the 2.0 behavior, which was a fixed rotation with no jitter
$transformer = new ImageRotator(offset: -90.0, jitter: 0.0);
```

!!! warning
    Models that relied on deterministic augmentation without jitter will now see randomized rotations unless you explicitly set `jitter: 0.0`.

### 38. GELU now uses the exact formula

The [GELU](neural-network/activation-functions/gelu.md) activation function was switched from the popular tanh approximation used in 2.0 to the exact Gaussian error function formula, and its derivative was replaced with the exact one:

```text
2.0 - tanh approximation
GELU(x)  = 0.5x (1 + tanh(√(2/π) (x + 0.044715x³)))

3.0 - exact
GELU(x)  = x Φ(x) = 0.5x (1 + erf(x / √2))
GELU'(x) = Φ(x) + x φ(x)
```

There is no API change — the constructor and the `activate()` and `differentiate()` methods are exactly as they were — but the outputs are not identical. The approximation tracks the exact curve closely without matching it, so a network with a GELU activation (inside an [Activation](neural-network/hidden-layers/activation.md) layer, for example) now trains on a slightly different function and may converge to different weights than the same model fit with 2.0. There is no flag to select the old approximation.

## New Features

The following changes are additive. They require no action to keep existing code working, but you can take advantage of them as part of your upgrade.

### 39. Parallelized nearest neighbors and Isolation Forest

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

### 40. Disk-based neural network snapshots

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

### 41. Clearable adaptive optimizer state

Optimizers such as [Adam](neural-network/optimizers/adam.md), [RMS Prop](neural-network/optimizers/rms-prop.md), [AdaGrad](neural-network/optimizers/adagrad.md), and [Momentum](neural-network/optimizers/momentum.md) maintain per-parameter state (gradient caches, momentum velocities) that is only needed during training. The neural network learners now expose a `cleanup()` method that discards this residual state by calling `flush()` on the optimizer — useful before reusing an estimator in a long-running process or to free memory after training:

```php
$mlp->train($dataset);

// Free residual optimizer state
$mlp->cleanup();
```

### 42. Validation interval for early stopping evaluation

The windowed gradient-based learners — MLP, [MLP Regressor](regressors/mlp-regressor.md), [Adaline](regressors/adaline.md), [Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), [Gradient Boost](regressors/gradient-boost.md), [AdaBoost](classifiers/adaboost.md), and [Logit Boost](classifiers/logit-boost.md) — now accept a `$evalInterval` constructor parameter (default `1`, or `3` for the boosting learners). It controls how often the validation set supplied via `setValidationDataset()` is scored during training, working in tandem with the `window` parameter for early stopping:

```php
$mlp = new MultilayerPerceptron(hiddenLayers: [new Dense(neurons: 100)], epochs: 1000, evalInterval: 5, window: 10);
```

### 43. Per-class and per-cluster variance smoothing

[Gaussian Naive Bayes](classifiers/gaussian-naive-bayes.md) and [Gaussian Mixture](clusterers/gaussian-mixture.md) now compute an independent variance epsilon for *each* class (or cluster) instead of a single global epsilon across all of them. This keeps fitting numerically stable even when classes or clusters have very different variance scales. There is no API change — the existing `$smoothing` parameter behaves as before, only the per-class application of it is new.

### 44. One Hot Encoder category exclusion

The [One Hot Encoder](transformers/one-hot-encoder.md) now accepts a list of `$ignoredCategories` to exclude from encoding. Categories in the list are skipped when the encoder is fitted, so they produce no columns. Only string and integer categories can be ignored:

```php
use Rubix\ML\Transformers\OneHotEncoder;

$encoder = new OneHotEncoder(['unknown', -1]); // ignore these categories
```

### 45. Class Purity and Cluster Purity metrics

Two new ground-truth clustering metrics were added — [Class Purity](cross-validation/metrics/class-purity.md) and [Cluster Purity](cross-validation/metrics/cluster-purity.md). They measure the extent to which each class (or cluster) is dominated by a single cluster (or class), returning a score between 0.0 and 1.0 where higher is better. They are complementary to the entropy-based [V-measure](cross-validation/metrics/v-measure.md), [Completeness](cross-validation/metrics/completeness.md), and [Homogeneity](cross-validation/metrics/homogeneity.md) metrics, and are only compatible with clusterers.

### 46. Float Type Converter

The new [Float Type Converter](transformers/float-type-converter.md) transformer converts integer and numeric-string values to their floating point equivalents. It is the drop-in remedy for the integers-as-categorical change in [item 1](#1-integers-are-now-a-categorical-data-type) — apply it to a dataset directly or add it to a Pipeline so that numeric features are always presented to the estimator as continuous:

```php
use Rubix\ML\Transformers\FloatTypeConverter;

$dataset->apply(new FloatTypeConverter());
```

### 47. Gradient accumulation and clipping for MLP learners

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

### 48. Layer freezing for fine-tuning

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

### 49. Dataset chunked() factory for online training

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

### 50. The progress() method

Learners, estimators, and transformers that record their progress epoch by epoch during training or transformation expose a `progress()` method that returns an iterable table combining every recorded epoch — the loss, the validation score (when a validation dataset was supplied), and, for neural network learners, the gradient norm — into a single ordered sequence:

```php
use Rubix\ML\Extractors\CSV;

$estimator->train($dataset);

$extractor = new CSV('progress.csv', true);

$extractor->export($estimator->progress());
```

Most iterative learners — including [Adaline](regressors/adaline.md), [MLP](classifiers/multilayer-perceptron.md), [Logistic Regression](classifiers/logistic-regression.md), [K Means](clusterers/k-means.md), [t-SNE](transformers/t-sne.md), and the others listed in [item 15](#15-the-steps-method-was-renamed-to-progress) — now expose this method. Because `progress()` returns a `Generator`, it can be streamed to an exporter or plotted without loading the whole table into memory. Learners that already exposed this table via `steps()` (renamed to `progress()` in [item 15](#15-the-steps-method-was-renamed-to-progress)) continue to work unchanged.

### 51. Grid Search fromNamedParams() factory method

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

### 52. Grid Search results() table

[Grid Search](grid-search.md) now generates a `results()` table: a Report of every parameter combination tested, in the order the trials were trained. Each row pairs the validation score (keyed by the metric name) with a nested `params` map of the combination's constructor parameters. It is convenient for inspecting the entire search space at a glance or exporting it:

```php
use Rubix\ML\Extractors\CSV;

$estimator->train($dataset);

$extractor = new CSV('results.csv', true);

$extractor->export($estimator->results());
```

The `best()` method has been removed; `scores()` is unchanged and continues to return the raw scores in trial order. `results()` throws a `RuntimeException` if no trials have been run yet.

### 53. Binned stratification for continuous labels

The [Labeled](datasets/labeled.md) dataset gained three methods that stratify a *continuous* label by **bin** rather than by exact label equality. Bin edges are derived from the quantiles of the label, so each bin holds roughly the same number of samples, and stratifying over them preserves the *shape* of the target distribution in every subset:

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::build($samples, [1.5, 2.75, 3.1, 4.9]);

// group samples into equal frequency bins
$strata = $dataset->stratifyByLabelBins(5);

// split into two subsets that each span the full range of the target
[$training, $testing] = $dataset->binnedSplit(0.8);

// or fold into k subsets with the same property
$folds = $dataset->binnedFold(5);
```

`stratifyByLabelBins($bins = 10)` returns one [Labeled](datasets/labeled.md) dataset per bin. `binnedSplit($ratio = 0.5, $bins = 10)` returns a left/right pair, and `binnedFold($k = 10, $bins = 10)` returns *k* subsets. Like their categorical counterparts, the split and fold variants return subsets that are equal in size and preserve the bin proportions; `binnedSplit()` additionally guarantees the left subset holds exactly `floor($ratio * numSamples())` samples.

There are a few constraints worth knowing about, since the bin count is a new degree of freedom the categorical methods don't have:

- **These methods reject categorical labels.** Passing a categorical dataset to `binnedSplit()` or `binnedFold()` throws an `InvalidArgumentException` — use `stratifiedSplit()` and `stratifiedFold()` there. The inverse also holds: `stratifyByClassLabels()`, `stratifiedSplit()`, and `stratifiedFold()` throw on a continuous label (see [item 17](#17-stratification-methods-were-renamed-and-now-require-categorical-labels)).
- **The bin count is capped automatically.** More bins means a tighter match to the target distribution, but each subset must be able to draw at least one sample from every bin. `binnedSplit()` reduces the requested count to at most `floor(numSamples() / ceil(1 / $ratio))` and `binnedFold()` to at most `floor(numSamples() / $k)`, so a small subset never loses the lowest or highest valued bin.
- **A `$bins` below 1 throws** an `InvalidArgumentException`, as does a `$ratio` outside of `[0.0, 1.0]` or a `$k` below 2. Like `stratifiedFold()`, `binnedFold()` throws when `$k` is greater than the number of samples in the smallest bin.
- **A constant target is fine.** If every label is the same value there is only one non-empty bin, and the split still divides the dataset evenly.

!!! note
    Empty bins are dropped from the result rather than returned as empty datasets, and a dataset with a constant target yields a single stratum. Because bins are ordered by target value, `binnedSplit()` shuffles the bins before awarding the leftover samples, so ties don't consistently favor the lowest valued bins. See [Binned Stratification](datasets/labeled.md#binned-stratification) for the full method reference.

### 54. Validators now stratify continuous labels by bin

The [Validators](cross-validation.md) now use the new binned stratification from [item 53](#53-binned-stratification-for-continuous-labels) whenever the label is continuous, instead of dividing the dataset at random. They dispatch on the label type, so categorical datasets are unaffected and keep using the existing categorical methods.

- [Hold Out](cross-validation/hold-out.md) and [Monte Carlo](cross-validation/monte-carlo.md) call `binnedSplit()` on continuous labels (previously `randomize()->split()` and `split()`).
- [K Fold](cross-validation/k-fold.md) calls `binnedFold()` on continuous labels (previously `fold()`).

The subset *sizes* are unchanged — the hold-out and fold partitions are still exactly as large as before. What changes is *which* samples land in each subset: a hold-out set drawn at random no longer tracks the shape of the target distribution, so it could end up with almost no samples from one end of the label range. The subsets are now spread evenly across that range, which is what makes them a meaningful validation signal.

!!! warning
    Because the hold-out membership differs from 2.0, models fit through [Hold Out](cross-validation/hold-out.md) or [Monte Carlo](cross-validation/monte-carlo.md) may differ from 2.0 even with identical hyper-parameters. The same applies to the validation set you inject into the windowed learners for early stopping — stratify it yourself if you need it to track the target distribution. See [item 57](#57-early-stopping-is-opt-in-via-user-supplied-validation-sets).

### 55. Binned description for continuous labels

[Labeled](datasets/labeled.md) gained `describeByLabelBins($bins = 10)`, the continuous-label counterpart to the `describeByClassLabels()` introduced in [item 17](#17-stratification-methods-were-renamed-and-now-require-categorical-labels). It reuses the equal frequency bins from [item 53](#53-binned-stratification-for-continuous-labels) to produce a [Report](cross-validation/reports/api.md#report-objects) describing the features of the dataset broken down by bin of the target:

```php
use Rubix\ML\Datasets\Labeled;

$dataset = Labeled::build($samples, [1.5, 2.75, 3.1, 4.9]);

$report = $dataset->describeByLabelBins(5);
```

In 2.0 there was no way to break a dataset down by a continuous target — `describeByLabel()` grouped samples by exact label equality, so a float target produced one stratum per distinct value. Comparing feature distributions across the range of a regression target now takes one call instead of a hand-rolled loop over `stratifyByLabel()`.

There is one difference from `describeByClassLabels()` worth knowing about: the report is a **list keyed by bin ordinal**, not a map keyed by name, because a bin has no intrinsic name. Bin 0 holds the lowest valued samples and the last key holds the highest. Empty bins are dropped, the bin count is reduced to at most `numSamples()`, a categorical label throws an `InvalidArgumentException`, and as with `describe()`, the label itself is included as the last column of every bin.

!!! note
    See [Describe by Label](datasets/labeled.md#describe-by-label) for the full method reference.

### 56. Image Rotator fill color

The [Image Rotator](transformers/image-rotator.md) gained a third constructor parameter, `$fillColor`, which controls the color used to fill the area exposed by rotation. In 2.0 the fill color was a hardcoded constant, so the exposed corners could not be controlled. The parameter accepts a 6-digit hex color string (with or without a leading `#`) and defaults to black (`'#000000'`):

```php
use Rubix\ML\Transformers\ImageRotator;

// fill the exposed area with white
$transformer = new ImageRotator(0.0, 0.2, '#ffffff');

// restore the 2.0 default
$transformer = new ImageRotator(0.0, 0.2); // fillColor = '#000000'
```

Any string that isn't a valid 6-digit hex color throws an `InvalidArgumentException` at construction. GD treats the background argument of `imagerotate()` as an RGB value for truecolor images but as a palette index for palette images, so the transformer allocates the color against the image when necessary — the same hex value works for both. This parameter is appended after `$jitter`, so existing calls that pass `$offset` and `$jitter` positionally are unaffected.

### 57. Early stopping is opt-in via user-supplied validation sets

The windowed learners no longer reserve a portion of the training set for validation. In 2.0, and throughout the 3.0 release candidates, an internal split meant that early stopping came at the cost of training on less data, and that the split itself was neither inspectable nor reusable. Now every one of these learners trains on the *entire* dataset given to `train()`, and the validation set is yours to supply.

[Adaline](regressors/adaline.md), [MLP Regressor](regressors/mlp-regressor.md), [Gradient Boost](regressors/gradient-boost.md), [Logistic Regression](classifiers/logistic-regression.md), [Softmax Classifier](classifiers/softmax-classifier.md), [Multilayer Perceptron](classifiers/multilayer-perceptron.md), [AdaBoost](classifiers/adaboost.md), and [Logit Boost](classifiers/logit-boost.md) all take the validation set through `setValidationDataset()`:

```php
use Rubix\ML\Classifiers\MultilayerPerceptron;
use Rubix\ML\Datasets\Labeled;

[$validation, $training] = Labeled::build($samples, $labels)->split(0.8);

$mlp = new MultilayerPerceptron(hiddenLayers: [$layer]);

// validate against an external split and train on all of $training
$mlp->setValidationDataset($validation);

$mlp->train($training);
```

With a validation set in place, the learner scores it every `$evalInterval` epochs and stops early when the score fails to improve for `$window` evaluations. Passing `null` disables progress monitoring and early stopping, leaving `scores()` empty.

There are a few constraints worth knowing about:

- **The set must be labeled and non-empty.** An empty dataset throws an `EmptyDataset` at the time of the call.
- **Dimensionality must match the training set.** A validation set whose `numFeatures()` differs from the training set throws an `IncorrectDatasetDimensionality` when training begins.
- **Classifiers require known labels.** Every label in the validation set must be one the classifier can emit, otherwise the score would be measured against classes the model can never predict. A validation set carrying an unknown label throws an `InvalidArgumentException`.
- **Online learners retain the set.** For the learners that implement [Online](online.md), the injected validation set persists across `partial()` calls instead of being re-derived from each incoming batch.
- **The set is transient.** Like the snapshot path, it is excluded from serialization. A learner restored from disk therefore has no validation set, so progress monitoring and early stopping stay inactive until you set one again.

### 58. PersistentTransformer decorator

The new [Persistent Transformer](transformers/persistent-transformer.md) decorator gives any [Stateful](transformers/api.md#stateful) transformer the `save()` and `load()` methods it needs to round-trip through storage. It interfaces with a [Persister](persisters/api.md) — the [Filesystem](persisters/filesystem.md) persister is the usual choice — and serializes with the [RBX](serializers/rbx.md) serializer unless you pass another one:

```php
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\HotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Persisters\Filesystem;

$transformer = new PersistentTransformer(
    new Pipeline([
        new HotDeckImputer(5),
        new OneHotEncoder(),
    ]),
    new Filesystem('pipeline.rbx')
);

$transformer->fit($dataset);

$transformer->save();
```

```php
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Persisters\Filesystem;

$transformer = PersistentTransformer::load(new Filesystem('pipeline.rbx'));
```

[Pipelines](transformers/pipeline.md) are the most common thing to decorate, and this is the migration path for the transformer state that 2.0 saved implicitly inside a persisted estimator (see [item 18](#18-pipeline-is-now-a-transformer-decorator)). Any other persistable Stateful transformer may be decorated just as well. Because the decorator delegates to the very same transformer instance it was constructed with, any fitting performed through it is captured by the next `save()`, and it can be handed straight to [apply()](datasets/api.md) like any other transformer.

There are a few constraints worth knowing about:

- **The decorator is a runtime object.** Unlike the transformer it decorates, it is not [Persistable](persistable.md) and therefore cannot itself be serialized. The [Persister](persisters/api.md) and [Serializer](serializers/api.md) it carries never leak into the transformer's saved state.
- **Restoring requires a Stateful base.** `load()` throws an `InvalidArgumentException` if the persisted object is not Stateful, and `save()` throws a `RuntimeException` if it is not Persistable.
- **`update()` must reach an Elastic base.** Calling `update()` when the decorated transformer is not [Elastic](transformers/api.md#elastic) throws a `RuntimeException`.

```php
$transformer->base()->update($dataset);
```

### 59. Grid Search setup() hook

[Grid Search](grid-search.md) gained a `setup()` method that registers a callback to invoke on every base estimator instance before it is cross-validated, and once more on the winning estimator before it is trained on the full dataset. It is the hook for configuration that has no constructor argument — attaching the early stopping validation set from [item 57](#57-early-stopping-is-opt-in-via-user-supplied-validation-sets), for example:

```php
use Rubix\ML\GridSearch;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\KFold;

$estimator = new GridSearch(LogisticRegression::class, $params, new FBeta(), new KFold(5));

$estimator->setup(function (LogisticRegression $learner) use ($testing) : void {
    $learner->setValidationDataset($testing);
});
```

The callback receives the fully constructed estimator and may call any of its public methods, whatever the base learner happens to expose. A few things worth knowing about:

- **It fires once per combination, plus once at the end.** Each candidate is configured before the validator scores it, and the best combination is configured again before its final training run on the full dataset.
- **It is transient.** The callback is excluded from serialization, so a Grid Search restored from disk comes back without one — register it again before training.

See [Setup](grid-search.md#setup) for the full reference.
