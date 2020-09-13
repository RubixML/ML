<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Graph/Trees/KDTree.php">[source]</a></span>

# K-d Tree
A multi-dimensional binary spatial tree for fast nearest neighbor queries. The K-d tree construction algorithm separates data points into bounded hypercubes or *boxes* that are used to determine which branches to prune off during nearest neighbor and range searches enabling them to complete in sub-linear time.

**Interfaces:** Binary Tree, Spatial

**Data Type Compatibility:** Continuous

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max leaf size | 30 | int | The maximum number of samples that each leaf node can contain. |
| 2 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |

## Example
```php
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Kernels\Distance\Euclidean;

$tree = new KDTree(30, new Euclidean());
```

## Additional Methods
This tree does not have any additional methods.

### References
>- J. L. Bentley. (1975). Multidimensional Binary Search Trees Used for Associative Searching.