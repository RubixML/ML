<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/TSNE.php">[source]</a></span>

# t-SNE

*T-distributed Stochastic Neighbor Embedding* is a two-stage non-linear manifold learning algorithm based on Batch Gradient Descent that seeks to maintain the distances between samples in low-dimensional space. During the first stage (*early stage*) the distances are exaggerated to encourage more pronounced clusters. Since the t-SNE cost function (KL Divergence) has a rough gradient, momentum is employed to help escape bad local minima.

!!! note
    T-SNE is implemented using the *exact* method which scales quadratically in the number of samples. Therefore, it is recommended to subsample datasets larger than a few thousand samples.

**Interfaces:** [Transformer](../transformers/api.md#transformer), [Iterative](../iterative.md), [Verbose](../verbose.md)

**Data Type Compatibility:** Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | dimensions | 2 | int | The number of dimensions of the target embedding. |
| 2 | rate | 100.0 | float | The learning rate that controls the global step size. |
| 3 | perplexity | 30 | int | The number of effective nearest neighbors to refer to when computing the variance of the distribution over that sample. |
| 4 | exaggeration | 12.0 | float | The factor to exaggerate the distances between samples during the early stage of embedding. |
| 5 | epochs | 1000 | int | The maximum number of times to iterate over the embedding. |
| 6 | minGradient | 1e-7 | float | The minimum norm of the gradient necessary to continue embedding. |
| 7 | evalInterval | 50 | int | The number of epochs to wait between evaluations of the KL Divergence cost. |
| 8 | window | 5 | int | The number of consecutive cost evaluations without improving on the best cost observed before early stopping. Set to 0 to disable early stopping. |
| 9 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between sample points. |

## Example

```php
use Rubix\ML\Transformers\TSNE;

$transformer = new TSNE(3, 10.0, 30, 12.0, 500, 1e-6);
```

## Additional Methods

Return the magnitudes of the gradient at each epoch from the last embedding.

```php
public norms() : float[]|null
```

Return the KL Divergence cost at each evaluation epoch from the last embedding.

```php
public losses() : float[]|null
```

## References

[^1]: L. van der Maaten et al. (2008). Visualizing Data using t-SNE.
[^2]: L. van der Maaten. (2009). Learning a Parametric Embedding by Preserving Local Structure.
