<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Datasets/Generators/Hyperplane.php">[source]</a></span>

# Hyperplane
Generates a labeled dataset whose samples form a hyperplane in n-dimensional vector space and whose labels are continuous values drawn from a uniform random distribution between -1 and 1. When the number of coefficients is either 1, 2 or 3, the samples form points, lines, and planes respectively. Due to its linearity, Hyperplane is especially useful for testing linear regression models.

**Data Types:** Continuous

**Label Type:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | coefficients | [1, -1] | array | The *n* coefficients of the hyperplane where n is the dimensionality. |
| 2 | intercept | 0.0 | float | The y intercept term. |
| 3 | noise | 0.1 | float | The factor of gaussian noise to add to the data points. |

## Example
```php
use Rubix\ML\Datasets\Generators\Hyperplane;

$generator = new Hyperplane([0.1, 3, -5, 0.01], 150.0, 0.25);
```

## Additional Methods
This generator does not have any additional methods.
