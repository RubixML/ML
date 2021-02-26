<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/Conduit.php">[source]</a></span>

# Conduit
Conduits allow you to abstract a series of transformations into a single higher-order transformation.

!!! note
    This transformer modifies the input dataset during fitting. If you need to keep a *clean* dataset in memory, you can clone the dataset object before calling the method that consumes it.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Elastic](api.md#elastic), [Persistable](../persistable.md)

**Data Type Compatibility:** Depends on base transformers

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | transformers | | array | The series of transformers to apply. |

## Example
```php
use Rubix\ML\Transformers\Conduit;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\Transformers\TruncatedSVD;

$transformer = new Conduit([
    new TextNormalizer(),
    new WordCountVecorizer(10000),
    new TruncatedSVD(100),
]);
```

## Additional Methods
Return the list of underlying transformer instances:
```php
public transformers() : array
```

```php
[$normalizer, $vectorizer, $svd] = $transformer->transformers();
```
