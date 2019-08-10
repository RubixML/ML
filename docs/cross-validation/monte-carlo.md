<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/MonteCarlo.php">Source</a></span>

# Monte Carlo
 Monte Carlo cross validation or *repeated random subsampling* is a technique that averages the validation score of a learner over a user-defined number of simulations where the learner is trained and tested on random splits of the dataset.

**Interfaces:** [Validator](api.md#validator), [Parallel](#parallel)

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | simulations | 10 | int | The number of simulations i.e. random subsamplings of the dataset. |
| 2 | ratio | 0.2 | float | The ratio of samples to hold out for testing. |

### Example
```php
use Rubix\ML\CrossValidation\MonteCarlo;

$validator = new MonteCarlo(30, 0.1);
```