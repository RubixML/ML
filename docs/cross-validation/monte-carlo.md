### Monte Carlo
Repeated Random Subsampling or Monte Carlo cross validation is a technique that takes the average validation score over a user-supplied number of simulations (randomized splits of the dataset).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/MonteCarlo.php)

**Interfaces:** [Parallel](#parallel)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | simulations | 10 | int | The number of simulations to run i.e the number of tests to average. |
| 2 | ratio | 0.2 | float | The ratio of samples to hold out for testing. |
| 3 | stratify | false | bool | Should we stratify the dataset before splitting? |

**Example:**

```php
use Rubix\ML\CrossValidation\MonteCarlo;

$validator = new MonteCarlo(30, 0.1);
```