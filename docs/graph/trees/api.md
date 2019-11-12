# Tree
A tree is a graph data structure with a root node and descendents called *child* nodes that form a hierarchy. One of the constraints of this model is that each node can only have a single parent node.

### Grow a Tree
Insert a root node and recursively split the dataset until a terminating condition is met:
```php
public grow(Dataset $dataset) : void
```

**Example**

```php
use Rubix\ML\Datasets\Labeled;

// Import samples and labels

$dataset = new Labeled($samples, $labels);

$tree->grow($dataset);
```

### Properties
Return the height of the tree i.e. the number of levels:
```php
public height() : int
```

Is the tree bare?
```php
public bare() : bool
```

**Example**

```php
var_dump($tree->height());
var_dump($tree->bare());
```

```sh
int(10)

bool(false)
```

### Destroy the Tree
Remove the root node and its descendents from the tree:
```php
public destroy() : void
```

# Binary Tree
Binary Trees are trees that are made up of nodes that have a maximum of 2 children (left and right). In machine learning, they are used to recursively split a dataset such as to learn a decision tree (a type of rule-based model) or to provide efficient nearest neighbor queries.

### Balance
Return the balance factor of the tree. A balanced tree will have a factor of 0 whereas an imbalanced tree will either be positive or negative indicating the direction and degree of the imbalance.

```php
public balance() : int
```

**Example**

```php
var_dump($tree->balance());
```

```sh
int(-1)
```

# BST
Binary Search Trees (BSTs) are binary tree structures that provide a fast O(log n) search API. They are used to power machine learning algorithms such as decision trees.

### Search
Search the tree for a leaf node or return null if not found:
```php
public search(array $sample) : Leaf
```

# Spatial
Spatial trees are constructed in such a way as to maximize performance on spatial queries such as nearest neighbor or radius searches. They do so by embedding spatial metadata within each node that enable fast search and pruning.

### Nearest Neighbors
Run a k nearest neighbors search and return the samples, labels, and distances in a 3-tuple:
```php
public nearest(array $sample, int $k) : array;
```

**Example**

```php
[$samples, $labels, $distances] = $tree->nearest($sample, 5);
```

### Radius Query
Return all samples, labels, and distances within a given radius of a sample:
```php
public range(array $sample, float $radius) : array
```

**Example**

```php
[$samples, $labels, $distances] = $tree->range($sample, 25.0);
```

# Decision Tree
A learner that induces a hierarchy of conditional control statements (called decision *rules*) is called a Decision Tree. A Decision Tree can be visualized as a flowchart in which each internal node represents a binary comparison on a particular feature (ex. height < 1.8288 or height >= 1.8288), and each leaf node represents the final outcome of a decision.

### Search
Search the decision tree for a leaf node and return it:
```php
public search(array $sample) : ?Outcome
```

**Example**

```php
$node = $tree->search($sample);
```

### Feature Importances
Return an array indexed by feature column that contains the normalized importance score of that feature:
```php
public featureImportances() : array
```

**Example**

```php
var_dump($tree->featureImpartances());
```

```sh
array(3) {
  [0]=> float(0)
  [1]=> float(0.46581631418525)
  [2]=> float(0.53418368581475)
}
```

### Print Rules
Return a human readable text representation of the decision tree rules:
```php
public rules() : string
```

**Example**
```php
echo $tree->rules();
```

```sh
|--- Column_2 < 143.69695373
|---|--- Column_1 < 65.12118525
|---|---|--- Outcome=red Impurity=0
|---|--- Column_1 >= 65.12118525
|---|---|--- Outcome=green Impurity=0.0790621
|--- Column_2 >= 143.69695373
|---|--- Outcome=blue Impurity=0
```