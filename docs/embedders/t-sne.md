<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Embedders/TSNE.php">Source</a></span>

# t-SNE
*T-distributed Stochastic Neighbor Embedding* is a two-stage non-linear manifold learning algorithm based on batch Gradient Descent that seeks to maintain the distances between samples in low dimensional space. During the first stage (*early exaggeration*) the distances are exaggerated to encourage more pronounced clusters. Since the t-SNE cost function (KL Divergence) has a rough gradient, additional momentum is employed to help escape bad local minima.

**Interfaces:** [Verbose](../verbose.md)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | 2 | int | The number of dimensions of the target embedding. |
| 2 | perplexity | 30 | int | The number of effective nearest neighbors to refer to when computing the variance of the Gaussian over that sample. |
| 3 | exaggeration | 12.0 | float | The factor to exaggerate the distances between samples during the early stage of fitting. |
| 4 | rate | 100.0 | float | The learning rate that controls the step size. |
| 5 | epochs | 1000 | int | The maximum number of times to iterate over the embedding. |
| 6 | min gradient | 1e-7 | float | The minimum gradient necessary to continue embedding. |
| 7 | window | 10 | int | The number of epochs without improvement in the training loss to wait before considering an early stop. |
| 8 | kernel | Euclidean | object | The distance kernel to use when measuring distances between samples. |

### Additional Methods
Return the magnitudes of the gradient at each epoch from the last embedding:
```php
public steps() : array
```

### Example
```php
use Rubi\ML\Embedders\TSNE;
use Rubix\ML\Kernels\Manhattan;

$embedder = new TSNE(3, 30, 12., 10., 500, 1e-6, 10, new Manhattan());
```

### References
>- L. van der Maaten et al. (2008). Visualizing Data using t-SNE.
>- L. van der Maaten. (2009). Learning a Parametric Embedding by Preserving Local Structure.