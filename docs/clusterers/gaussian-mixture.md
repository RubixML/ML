<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/GaussianMixture.php">[source]</a></span>

# Gaussian Mixture
A Gaussian Mixture model (GMM) is a probabilistic model for representing the presence of clusters within an overall population without requiring a sample to know which sub-population it belongs to beforehand. GMMs are similar to centroid-based clusterers like [K Means](k-means.md) but allow both the cluster centers (*means*) as well as the radii (*variances*) to be learned as well. For this reason, GMMs are especially useful for clusterings that are of different radii.

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Probabilistic](../probabilistic.md), [Verbose](../verbose.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | | int | The number of target clusters. |
| 2 | epochs | 100 | int | The maximum number of training rounds to execute. |
| 3 | minChange | 1e-3 | float | The minimum change in the components necessary for the algorithm to continue training. |
| 4 | seeder | PlusPlus | Seeder | The seeder used to initialize the Gaussian components. |

## Example
```php
use Rubix\ML\Clusterers\GaussianMixture;
use Rubix\ML\Clusterers\Seeders\KMC2;

$estimator = new GaussianMixture(5, 100, 1e-4, new KMC2(50));
```

## Additional Methods
Return the cluster prior probabilities based on their representation over all training samples:
```php
public priors() : float[]
```

Return the running means of each feature column for each cluster:
```php
public means() : array[]
```

Return the variance of each feature column for each cluster:
```php
public variances() : array[]
```

Return the loss at each epoch from the last training session:
```php
public steps() : float[]|null
```

## References
[^1]: A. P. Dempster et al. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm.
[^2]: J. Blomer et al. (2016). Simple Methods for Initializing the EM Algorithm for Gaussian Mixture Models.