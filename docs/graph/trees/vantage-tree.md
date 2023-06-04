<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Graph/Trees/VPTree.php">[source]</a></span>

# Vantage Tree
A Vantage Point Tree is a binary spatial tree that divides samples by their distance from the center of a cluster called the *vantage point*. Samples that are closer to the vantage point will be put into one branch of the tree while samples that are farther away will be put into the other branch.

**Interfaces:** Binary Tree, Spatial

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | max leaf size | 30 | int | The maximum number of samples that each leaf node can contain. |
| 2 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |

## Example
```php
use Rubix\ML\Graph\Trees\VantageTree;
use Rubix\ML\Kernels\Distance\Euclidean;

$tree = new VantageTree(30, new Euclidean());
```

## Additional Methods
This tree does not have any additional methods.

### References
>- P. N. Yianilos. (1993). Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces.
