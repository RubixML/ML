<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Datasets/Generators/Agglomerate.php">[source]</a></span>

# Agglomerate
An Agglomerate is a collection of generators with each of them given a user-defined label. Agglomerates are useful for classification, clustering, and anomaly detection problems where the target label is a discrete value.

**Data Types:** Depends on base generators

**Label Type:** Categorical

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | generators | | array | A collection of generators indexed by their given label. |
| 2 | weights | Auto | array | A set of arbitrary weight values corresponding to a generator's proportion of the overall agglomeration. If no weights are given, each generator is assigned equal weight. |

## Example
```php
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\HalfMoon;
use Rubix\ML\Datasets\Generators\Circle;

$generator = new Agglomerate([
	'foo' => new Blob([5, 2], 1.0),
	'bar' => new HalfMoon(-3, 5, 1.5, 90.0, 0.1),
	'baz' => new Circle(2, -4, 2.0, 0.05),
], [
	3.5, 4.0, 5.0,
]);
```

## Additional Methods
Return the normalized weight values of each generator in the agglomerate:
```php
public weights() : array
```
