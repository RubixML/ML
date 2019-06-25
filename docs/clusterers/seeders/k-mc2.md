<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Clusterers/Seeders/KMC2.php">Source</a></span>

# K-MC2
This is a fast [Plus Plus](plus-plus.md) approximator that replaces the brute force method with a substantially faster Markov Chain Monte Carlo (MCMC) sampling method with comparable performance.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | m | 50 | int | The number of candidate nodes in the Markov Chain. |
| 2 | kernel | Euclidean | object | The distance kernel used to compute the distance between samples. |

### Example
```php
use Rubix\ML\Clusterers\Seeders\KMC2;
use Rubix\ML\Kernels\Distance\Euclidean;

$seeder = new KMC2(200, new Euclidean());
```

### References
>- O. Bachem et al. (2016). Approximate K-Means++ in Sublinear Time.