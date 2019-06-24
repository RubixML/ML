### Blurry Percentile
A strategy that guesses within the domain of the p-th percentile of the fitted data plus some gaussian noise.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/BlurryPercentile.php)

**Data Type:** Continuous

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | p | 50.0 | float | The index of the percentile to predict where 50 is the median. |
| 2 | blur | 0.1 | float | The amount of gaussian noise to add to the guess as a factor of the median absolute deviation (MAD). |

**Example:**

```php
use Rubix\ML\Other\Strategies\BlurryPercentile;

$strategy = new BlurryPercentile(34.0, 0.2);
```