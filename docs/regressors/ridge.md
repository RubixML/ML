<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Regressors/Ridge.php">[source]</a></span>

# Ridge
L2 regularized linear regression solved using a closed-form solution. The addition of regularization, controlled by the *alpha* hyper-parameter, makes Ridge less likely to overfit the training data than ordinary least squares (OLS).

**Interfaces:** [Estimator](../estimator.md), [Learner](../learner.md), [Ranks Features](../ranks-features.md), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | l2Penalty | 1.0 | float | The strength of the L2 regularization penalty. |

## Example
```php
use Rubix\ML\Regressors\Ridge;

$estimator = new Ridge(2.0);
```

## Additional Methods
Return the weights of features in the decision function.
```php
public coefficients() : array|null
```

Return the bias added to the decision function.
```php
public bias() : float|null
```
