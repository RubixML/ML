# Embedders
Embedders are non-linear dimensionality reducers that produce dense representations of an input feature space such that they can be vizualized or used as lower dimensional inputs to a learning algorithm.

### Embed a Dataset
To embed a dataset and return an array containing the low dimensional samples:
```php
public embed(Dataset $dataset) : array
```

**Example**

```php
use Rubix\ML\Datasets\Unlabeled;

// Import high dimensional samples

$dataset = new Unlabeled($samples);

$samples = $embedder->embed($dataset);
```