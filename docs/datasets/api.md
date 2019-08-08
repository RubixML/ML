# Dataset Objects
In Rubix, data are passed around using specialized container structures called Dataset objects. Dataset objects can hold a heterogeneous mix of categorical and continuous data and make it easy to transport data in a canonical way. 

> **Note:** There are two *types* of features that estimators can process i.e *categorical* and *continuous*. Any numerical (integer or float) datum is considered continuous and any string datum is considered categorical by convention throughout Rubix.

### Selecting
Return all the samples in the dataset:
```php
public samples() : array
```

Select the *sample* at row offset:
```php
public row(int $index) : array
```

Select the *values* of a feature column at offset:
```php
public column(int $index) : array
```

### Properties
Return the number of rows in the dataset:
```php
public numRows() : int
```

Return the number of columns in the dataset:
```php
public numColumns() : int
```

Return the integer encoded column types for each feature column:
```php
public types() : array
```

Return the integer encoded column type given a column index:
```php
public columnType(int $index) : int
```

### Applying Transformations
You can apply a [Transformer](#transformers) directly to a Dataset by passing it to the apply method on the Dataset.

```php
public apply(Transformer $transformer) : void
```

**Example**

```php
use Rubix\ML\Transformers\OneHotEncoder;

$dataset->apply(new OneHotEncoder());
```

### Stacking
Stack a number of dataset objects on top of each other to form a single dataset:
```php
public static stack(array $datasets) : self
```

### Prepending and Appending
To prepend a given dataset onto the beginning of another dataset:
```php
public prepend(Dataset $dataset) : self
```

To append a given dataset onto the end of another dataset:
```php
public append(Dataset $dataset) : self
```

### Head and Tail
Return the *first* **n** rows of data in a new dataset object:
```php
public head(int $n = 10) : self
```

Return the *last* **n** rows of data in a new dataset object:
```php
public tail(int $n = 10) : self
```

**Example**

```php
// Return the first 5 rows in a new dataset
$subset = $dataset->head(5);

// Return the last 10 rows in a new dataset
$subset = $dataset->tail(10);
```

### Taking and Leaving
Remove **n** rows from the dataset and return them in a new dataset:
```php
public take(int $n = 1) : self
```

Leave **n** samples on the dataset and return the rest in a new dataset:
```php
public leave(int $n = 1) : self
```

### Slicing and Splicing
Return an *n* size portion of the dataset in a new dataset:
```php
public slice(int $offset, int $n) : self
```

Remove a size *n* chunk of the dataset starting at *offset* and return it in a new dataset:
```php
public splice(int $offset, int $n) : self
```

# Splitting
Split the dataset into left and right subsets given by a *ratio*:
```php
public split(float $ratio = 0.5) : array
```

Partition the dataset into left and right subsets based on the value of a feature in a specified column:
```php
public partition(int $index, mixed $value) : array
```

**Example**

```php
// Split the dataset into left and right subsets
[$left, $right] = $dataset->split(0.5);

// Partition the dataset by the feature column at index 4 by value '50'
[$left, $right] = $dataset->partition(4, 50);
```

### Folding
Fold the dataset *k* - 1 times to form *k* equal size datasets:
```php
public fold(int $k = 10) : array
```

**Example**

```php
// Fold the dataset into 8 equal size datasets
$folds = $dataset->fold(8);

var_dump(count($folds));
```

**Output**

```sh
int(8)
```

### Batching
Batch the dataset into subsets containing a maximum of *n* rows per batch:
```php
public batch(int $n = 50) : array
```

### Randomization
Randomize the order of the Dataset and return it for method chaining:
```php
public randomize() : self
```

Generate a random subset without replacement:
```php
public randomSubset(int $n) : self
```

Generate a random subset with replacement of size *n*:
```php
public randomSubsetWithReplacement($n) : self
```

Generate a random *weighted* subset with replacement of size *n*:
```php
public randomWeightedSubsetWithReplacement($n, array $weights) : self
```

**Example**

```php
// Randomize and split the dataset and split into two subsets
[$left, $right] = $dataset->randomize()->split(0.8);

// Generate a random unique subset of 50 random samples
$subset = $dataset->randomSubset(50);

// Generate a 'bootstrap' dataset of 500 random samples
$subset = $dataset->randomSubsetWithReplacement(500);

// Sample a random subset according to a given weight distribution
$subset = $dataset->randomWeightedSubsetWithReplacement(200, $weights);
```

### Filtering
To filter a Dataset by a feature column:
```php
public filterByColumn(int $index, callable $fn) : self
```

**Example**

```php
$tallPeople = $dataset->filterByColumn(2, function ($value) {
	return $value > 178.5;
});
```

### Sorting
To sort a dataset in place by a specific feature column:
```php
public sortByColumn(int $index, bool $descending = false) : self
```

**Example**

```php
var_dump($dataset->samples());

$dataset->sortByColumn(2, false);

var_dump($dataset->samples());
```

**Output**

```sh
array(3) {
    [0]=> array(3) {
	    [0]=> string(4) "mean"
	    [1]=> string(4) "furry"
	    [2]=> int(8)
    }
    [1]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(1)
    }
    [2]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(6)
    }
}

array(3) {
    [0]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(1)
    }
    [1]=> array(3) {
	    [0]=> string(4) "nice"
	    [1]=> string(4) "rough"
	    [2]=> int(6)
    }
    [2]=> array(3) {
	    [0]=> string(4) "mean"
	    [1]=> string(4) "furry"
	    [2]=> int(8)
    }
}
```