### Embedders
Manifold learning is a type of non-linear dimensionality reduction used primarily for visualizing high dimensional datasets in low (1 to 3) dimensions. Embedders are manifold learners that embed high dimensional datasets into lower ones.

To embed a dataset and return the low dimensional dataset:
```php
public embed(Dataset $dataset) : Dataset
```

### Example
```php
use Rubix\ML\Datasets\Unlabeled;

// Import high dimensional samples

$high = new Unlabeled($samples);

$low = $embedder->embed($high);
```