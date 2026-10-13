<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/Trustworthiness.php">[source]</a></span>

# Trustworthiness

Trustworthiness measures how well a low-dimensional embedding (e.g. produced by t-SNE or PCA) preserves the local neighbourhood structure of the original feature space. For every sample, its nearest neighbours in the embedded space are inspected; any that were distant relatives in the input space are penalised in proportion to their rank. A perfect preservation of local structure yields a score of 1.0; the more the local ordering degrades, the lower the score.

$$
{\displaystyle T = 1 - \frac{2}{n\,k\,(2n - 3k - 1)} \sum_{i=1}^{n} \sum_{j \in \mathcal{N}^{k}_{i}} \max(0, r(i,j) - k)}
$$

where *n* is the number of samples, *k* is the number of neighbours, $\mathcal{N}^{k}_{i}$ is the set of *k* neighbours of sample *i* in the embedded space, and *r*(i, j) is the rank of sample *j* among *i*'s neighbours in the original space.

A score of 1.0 means the embedding preserves local structure perfectly; lower scores indicate that points which were close together in the original space have been separated (or far-apart points have been brought together) by the projection.

!!! note
    The number of neighbours *k* must be less than *n* / 2, otherwise the score is not guaranteed to lie within [0, 1].

**Estimator Compatibility:** None

**Score Range:** 0 to 1

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | nNeighbors | 5 | int | The number of nearest neighbours to consider when scoring local structure. |
| 2 | kernel | Euclidean | Distance | The kernel used to measure dissimilarity in the original feature space. |

## Example

```php
use Rubix\ML\Transformers\TSNE;
use Rubix\ML\Kernels\Distance\Euclidean;
use Rubix\ML\CrossValidation\Metrics\Trustworthiness;

$transformer = new TSNE(
    dimensions: 2,
    rate: 10.0,
    perplexity: 30,
    exaggeration: 12.0,
    epochs: 1000,
    minGradient: 1e-7,
    evalInterval: 50,
    window: 6,
);

$originalSamples = $dataset->samples();
$embedding = $originalSamples;

$transformer->transform($embedding);

$metric = new Trustworthiness(5, new Euclidean());

$score = $metric->score($originalSamples, $embedding);

echo $score;
```

```text
0.88
```

## References

[^1]: J. Venna and S. Kaski. (2001). Neighborhood Preservation in Nonlinear Projection Methods: An Experimental Study.
