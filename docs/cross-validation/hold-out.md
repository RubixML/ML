<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/HoldOut.php">[source]</a></span>

# Hold Out
Hold Out is a quick and simple cross validation technique that uses a validation set that is *held out* from the training data. The advantages of Hold Out is that the validation score is quick to compute, however it does not allow the learner to *both* train and test on all the data in the training set.

**Interfaces:** [Validator](api.md#validator)

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | ratio | 0.2 | float | The ratio of samples to hold out for testing. |

## Example
```php
use Rubix\ML\CrossValidation\HoldOut;

$validator = new HoldOut(0.3);
```