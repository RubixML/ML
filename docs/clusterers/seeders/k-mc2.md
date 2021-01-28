<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/Seeders/KMC2.php">[source]</a></span>

# K-MC2
A fast [Plus Plus](plus-plus.md) approximator that replaces the brute force method with a substantially faster Markov Chain Monte Carlo (MCMC) sampling procedure with comparable results.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | m | 50 | int | The number of candidate nodes in the Markov Chain. |
| 2 | kernel | Euclidean | Distance | The distance kernel used to compute the distance between samples. |

## Example
```php
use Rubix\ML\Clusterers\Seeders\KMC2;
use Rubix\ML\Kernels\Distance\Euclidean;

$seeder = new KMC2(200, new Euclidean());
```

###
[^1]: O. Bachem et al. (2016). Approximate K-Means++ in Sublinear Time.