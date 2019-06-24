### Ridge
L2 penalized least squares linear regression solved using closed-form equation.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Regressors/Ridge.php)

**Interfaces:** [Estimator](#estimators), [Learner](#learner), [Persistable](#persistable)

**Compatibility:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | alpha | 1.0 | float | The L2 regularization penalty amount to be added to the weight coefficients. |

**Additional Methods:**

Return the weights of the model:
```php
public weights() : array|null
```

Return the bias parameter:
```php
public bias() : float|null
```

**Example:**
```php
use Rubix\ML\Regressors\Ridge;

$estimator = new Ridge(2.0);
```