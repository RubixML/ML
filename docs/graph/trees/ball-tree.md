<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Graph/Trees/BallTree.php">[source]</a></span>

# Ball Tree
A binary spatial tree that partitions a dataset into successively smaller and tighter *ball* nodes whose boundaries are defined by a hypersphere. Ball Tree works well in higher dimensions since the partitioning schema does not rely on a finite number of 1-dimensional axis aligned splits such as with [k-d tree](k-d-tree.md).

**Interfaces:** Binary Tree, Spatial

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | maxLeafSize | 30 | int | The maximum number of samples that each leaf node can contain. |
| 2 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |

## Example
```php
use Rubix\ML\Graph\Trees\BallTree;
use Rubix\ML\Kernels\Distance\Euclidean;

$tree = new BallTree(40, new Euclidean());
```

## Additional Methods
This tree does not have any additional methods.

## References
[^1]: S. M. Omohundro. (1989). Five Balltree Construction Algorithms.
[^2]: M. Dolatshah et al. (2015). Ball*-tree: Efficient spatial indexing for constrained nearest-neighbor search in metric spaces.