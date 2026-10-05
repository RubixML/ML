<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/Pipeline.php">[source]</a></span>

# Pipeline

Pipeline is a [Transformer](api.md) decorator capable of composing an arbitrarily long series of Transformer middleware into a single unit. It fits the stack to a training dataset without mutating it, transforms incoming samples by streaming them through each transformer in order, and — when updated — refines the fitting of [Elastic](api.md#elastic) transformers (or lazily fits any [Stateful](api.md#stateful) ones that have not yet been seen) while streaming a working copy of the data through the chain.

**Interfaces:** [Transformer](api.md), [Stateful](api.md#stateful), [Elastic](api.md#elastic), [Persistable](../persistable.md)

**Data Type Compatibility:** Defined by the first transformer in the stack; an empty pipeline accepts every data type

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | transformers | | array | A list of transformers to be composed in order. |

## Example

```php
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\HotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\ZScaleStandardizer;

$transformer = new Pipeline([
    new HotDeckImputer(5),
    new OneHotEncoder(),
    new ZScaleStandardizer(),
]);
```

## Fitting and Updating

Because Pipeline is a [Stateful](api.md#stateful) transformer itself, it exposes `fit()` and `fitted()`. Calling `fit()` will refit every stateful transformer in the stack to the current dataset, streaming a working copy of the data through the chain — the input dataset is left unaltered.

```php
$transformer = new Pipeline([
    new OneHotEncoder(),
    new ZScaleStandardizer(),
]);

$transformer->fit($dataset);
```

If any transformer in the stack is [Elastic](api.md#elastic), the pipeline is also Elastic: `update()` will refine each elastic fitting and lazily fit any stateful transformer that has not yet been seen, again streaming a working copy of the data through the chain without touching the input.

```php
$transformer->update($dataset);
```

Transformers that are stateless and non-elastic are applied as-is — they are always `fitted()`, so they contribute nothing to the `fitted()` status of the pipeline. An empty pipeline is always considered fitted and behaves as a no-op.

Since fitting does not transform the data, transform your dataset explicitly with the pipeline's `transform()` method or, more idiomatically, with `Dataset::apply()`:

```php
$dataset->apply($transformer);
```

## Persistence

A fitted pipeline can be saved to and loaded from storage by decorating it with the [Persistent Transformer](persistent-transformer.md), which interfaces with the persistence subsystem on your behalf.

```php
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Persisters\Filesystem;

$transformer = new PersistentTransformer(
    new Pipeline([new OneHotEncoder()]),
    new Filesystem('example.rbx')
);

$transformer->fit($dataset);

$transformer->save();
```

```php
$transformer = PersistentTransformer::load(new Filesystem('example.rbx'));
```
