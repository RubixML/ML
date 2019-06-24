### Constant
Always guess a constant value.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/Constant.php)

**Data Type:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | value | 0. | float | The value to guess. |

**Example:**
```php
use Rubix\ML\Other\Strategies\Constant;

$strategy = new Constant(17.);
```