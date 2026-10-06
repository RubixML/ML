<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/PersistentTransformer.php">[source]</a></span>

# Persistent Transformer

The Persistent Transformer decorator wraps a [Stateful](api.md#stateful) transformer with additional functionality for saving and loading its fitting. It uses [Persister](../persisters/api.md) objects to interface with various storage backends such as the [Filesystem](../persisters/filesystem.md). [Pipelines](pipeline.md) are the most common use case, but any [Persistable](../persistable.md) Stateful transformer may be decorated.

**Interfaces:** [Transformer](api.md), [Stateful](api.md#stateful), [Elastic](api.md#elastic)

**Data Type Compatibility:** Depends on base transformer

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | base | | Stateful | The persistable base transformer. |
| 2 | persister | | Persister | The persister used to interface with the storage system. |
| 3 | serializer | RBX | Serializer | The object serializer. |

## Examples

```php
use Rubix\ML\Transformers\Pipeline;
use Rubix\ML\Transformers\HotDeckImputer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Persisters\Filesystem;

$transformer = new PersistentTransformer(
    new Pipeline([
        new HotDeckImputer(5),
        new OneHotEncoder(),
    ]),
    new Filesystem('example.rbx')
);
```

Any Stateful transformer can be decorated as well.

```php
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Persisters\Filesystem;

$transformer = new PersistentTransformer(
    new ZScaleStandardizer(),
    new Filesystem('standardizer.rbx')
);
```

Because the decorator delegates to the very same transformer instance it was constructed with, any fitting performed through it is captured by the next call to `save()`. The decorator behaves like the transformer it wraps everywhere else, so it can be passed directly to `Dataset::apply()`.

```php
$transformer->fit($dataset);

$transformer->save();
```

## Additional Methods

Load the transformer from storage.

```php
public static load(Persister $persister, ?Serializer $serializer = null) : self
```

```php
use Rubix\ML\Transformers\PersistentTransformer;
use Rubix\ML\Persisters\Filesystem;

$transformer = PersistentTransformer::load(new Filesystem('example.rbx'));
```

Save the transformer and the state of its fitting to storage.

```php
public save() : void
```

```php
$transformer->save();
```

Return the base transformer instance.

```php
public base() : Stateful
```

```php
$transformer = $transformer->base();
```

## Caveats

Calling `update()` on a base transformer that does not implement the [Elastic](api.md#elastic) interface will throw a RuntimeException, in which case it must be called on the base transformer directly instead.

```php
$transformer->base()->update($dataset);
```

Unlike the transformer it decorates, the decorator is not [Persistable](../persistable.md). It cannot itself be serialized, and the storage coordinates it carries are never a part of the transformer's saved state.