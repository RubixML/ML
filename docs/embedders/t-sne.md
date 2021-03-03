<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Embedders/TSNE.php">[source]</a></span>

# t-SNE
*T-distributed Stochastic Neighbor Embedding* is a two-stage non-linear manifold learning algorithm based on Batch Gradient Descent that seeks to maintain the distances between samples in low-dimensional space. During the first stage (*early stage*) the distances are exaggerated to encourage more pronounced clusters. Since the t-SNE cost function (KL Divergence) has a rough gradient, momentum is employed to help escape bad local minima.

!!! note
    T-SNE is implemented using the *exact* method which scales quadratically in the number of samples. Therefore, it is recommended to subsample datasets larger than a few thousand samples.

**Interfaces:** [Transformer](../transformers/api.md#transformer), [Verbose](../verbose.md)

**Data Type Compatibility:** Depends on distance kernel

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | 2 | int | The number of dimensions of the target embedding. |
| 2 | rate | 100.0 | float | The learning rate that controls the global step size. |
| 3 | perplexity | 30 | int | The number of effective nearest neighbors to refer to when computing the variance of the distribution over that sample. |
| 4 | exaggeration | 12.0 | float | The factor to exaggerate the distances between samples during the early stage of embedding. |
| 5 | epochs | 1000 | int | The maximum number of times to iterate over the embedding. |
| 6 | minGradient | 1e-7 | float | The minimum norm of the gradient necessary to continue embedding. |
| 7 | window | 10 | int | The number of epochs without improvement in the training loss to wait before considering an early stop. |
| 8 | kernel | Euclidean | Distance | The distance kernel to use when measuring distances between samples. |

## Example
```php
use Rubix\ML\Embedders\TSNE;
use Rubix\ML\Kernels\Distance\Manhattan;

$embedder = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6, 10, new Manhattan());
```

## Additional Methods
Return the magnitudes of the gradient at each epoch from the last embedding:
```php
public steps() : float[]|null
```

## References
[^1]: L. van der Maaten et al. (2008). Visualizing Data using t-SNE.
[^2]: L. van der Maaten. (2009). Learning a Parametric Embedding by Preserving Local Structure.