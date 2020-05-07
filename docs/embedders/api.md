# Embedders
Embedders are dimensionality reducers that produce dense feature representations from the samples contained in a dataset. They are a type of [Transformer](../transformers/api.md) that work more like a Learner under the hood. Embedders are typically used for visualizing high-dimensional datasets or for producing inputs to a learning algorithm.

### Embed a Dataset
To embed a dataset you can use the `apply()` method on a dataset object like you would a regular [Transformer](../transformers/api.md).

**Example**

```php
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Embedders\TSNE;

$dataset = new Unlabeled($samples);

echo $dataset->numColumns(); // 100

$dataset->apply(new TSNE(3));

echo $dataset->numColumns(); // 3
```