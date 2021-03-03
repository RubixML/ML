<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Graph/Trees/KDTree.php">[source]</a></span>

# K-d Tree
A multi-dimensional binary spatial tree for fast nearest neighbor queries. The K-d tree construction algorithm separates data points into bounded hypercubes or *boxes* that are used to determine which branches to prune off during nearest neighbor and range searches enabling them to complete in sub-linear time.

**Interfaces:** Binary Tree, Spatial

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | maxLeafSize | 30 | int | The maximum number of samples that each leaf node can contain. |
| 2 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |

## Example
```php
use Rubix\ML\Graph\Trees\KDTree;
use Rubix\ML\Kernels\Distance\Euclidean;

$tree = new KDTree(30, new Euclidean());
```

## Additional Methods
This tree does not have any additional methods.

## References
[^1]: J. L. Bentley. (1975). Multidimensional Binary Search Trees Used for Associative Searching.