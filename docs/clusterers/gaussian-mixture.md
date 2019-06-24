<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Clusterers/GaussianMixture.php">Source</a></span></p>

# Gaussian Mixture
A Gaussian Mixture model (GMM) is a probabilistic model for representing the presence of clusters within an overall population without requiring a sample to know which sub-population it belongs to a priori. GMMs are similar to centroid-based clusterers like [K Means](#k-means) but allow both the centers (*means*) *and* the radii (*variances*) to be learned as well.

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Probabilistic](#probabilistic), [Verbose](#verbose), [Persistable](#persistable)

**Data Type Compatibility:** Continuous

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | | int | The number of target clusters. |
| 2 | epochs | 100 | int | The maximum number of training rounds to execute. |
| 3 | min change | 1e-3 | float | The minimum change in the components necessary for the algorithm to continue training. |
| 6 | seeder | PlusPlus | object | The seeder used to initialize the Guassian components. |

### Additional Methods
Return the cluster prior probabilities based on their representation over all training samples:
```php
public priors() : array
```

Return the running means of each feature column for each cluster:
```php
public means() : array
```

Return the variance of each feature column for each cluster:
```php
public variances() : array
```

### Example
```php
use Rubix\ML\Clusterers\GaussianMixture;
use Rubix\ML\Clusterers\Seeders\KMC2;

$estimator = new GaussianMixture(5, 1e-4, 100, new KMC2(50));
```

### References
>- A. P. Dempster et al. (1977). Maximum Likelihood from Incomplete Data via the EM Algorithm.
>- J. Blomer et al. (2016). Simple Methods for Initializing the EM Algorithm for Gaussian Mixture Models.