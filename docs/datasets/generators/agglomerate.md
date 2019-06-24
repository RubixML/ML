<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Datasets/Generators/Agglomerate.php">Source</a></span></p>

# Agglomerate
An Agglomerate is a collection of generators with each of them given a user-defined label. Agglomerates are useful for classification, clustering, and anomaly detection problems where the target label is a discrete value.

**Data Types:** Depends on base generators

**Label Type:** Categorical

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | generators | | array | A collection of generators indexed by their given label. |
| 2 | weights | Auto | array | A set of arbitrary weight values corresponding to a generator's proportion of the overall agglomeration. |

### Additional Methods
Return the normalized weight values of each generator in the agglomerate:
```php
public weights() : array
```

### Example
```php
use Rubix\ML\Datasets\Generators\Agglomerate;

$generator = new Agglomerate([
	new Blob([5, 2], 1.0),
	new HalfMoon(-3, 5, 1.5, 90.0, 0.1),
	new Circle(2, -4, 2.0, 0.05),
], [
	3, 4, 5, // Arbitrary weight values
]);
```