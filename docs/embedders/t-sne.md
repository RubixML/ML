### t-SNE
*T-distributed Stochastic Neighbor Embedding* is a two-stage non-linear manifold learning algorithm based on batch Gradient Descent. During the first stage (*early* stage) the samples are exaggerated to encourage distant clusters. Since the t-SNE cost function (KL Divergence) has a rough gradient, momentum is employed to help escape bad local minima.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Embedders/TSNE.php)

**Interfaces:** [Verbose](#verbose)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | dimensions | 2 | int | The number of dimensions of the target embedding. |
| 2 | perplexity | 30 | int | The number of effective nearest neighbors to refer to when computing the variance of the Gaussian over that sample. |
| 3 | exaggeration | 12. | float | The factor to exaggerate the distances between samples during the early stage of fitting. |
| 4 | rate | 100. | float | The learning rate that controls the step size. |
| 5 | kernel | Euclidean | object | The distance kernel to use when measuring distances between samples. |
| 6 | epochs | 1000 | int | The number of times to iterate over the embedding. |
| 7 | min gradient | 1e-8 | float | The minimum gradient necessary to continue embedding. |
| 8 | window | 3 | int | The number of most recent epochs to consider when determining an early stop. |

**Additional Methods:**

Return the magnitudes of the gradient at each epoch from the last embedding:
```php
public steps() : array
```

**Example:**

```php
use Rubi\ML\Embedders\TSNE;
use Rubix\ML\Kernels\Manhattan;

$embedder = new TSNE(2, 30, 12., 10., new Manhattan(), 500, 1e-6, 5);
```

**References:**

>- L. van der Maaten et al. (2008). Visualizing Data using t-SNE.
>- L. van der Maaten. (2009). Learning a Parametric Embedding by Preserving Local Structure.