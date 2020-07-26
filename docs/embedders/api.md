# Embedders
Embedders are dimensionality reducers that produce dense feature representations from the samples contained in a dataset. They are a type of [Transformer](../transformers/api.md) but they operate more like a Learner under the hood. In fact, the type of learning that Embedders employ is called *manifold* learning because it aims to find a low-dimensional manifold of a high-dimensional dataset. Embedders are typically used for visualizing high-dimensional datasets and for producing inputs to a learning algorithm.

### Embed a Dataset
To embed a dataset you can use the `apply()` method on a dataset object like you would a regular [Transformer](../transformers/api.md).

```php
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Embedders\TSNE;

$dataset = new Unlabeled($samples);

echo $dataset->numColumns(); // 100 dimensions

$dataset->apply(new TSNE(3, 200.0));

echo $dataset->numColumns(); // 3 dimensions
```