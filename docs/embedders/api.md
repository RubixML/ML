# Embedders
Embedders are non-linear dimensionality reducers that produce dense representations of an input feature space such that they can be vizualized or used as inputs to a learning algorithm.

### Embed a Dataset
To embed a dataset and return the low dimensional dataset:
```php
public embed(Dataset $dataset) : Dataset
```

**Example**

```php
use Rubix\ML\Datasets\Unlabeled;

// Import high dimensional samples

$high = new Unlabeled($samples);

$low = $embedder->embed($high);
```