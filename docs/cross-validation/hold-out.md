<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/HoldOut.php">Source</a></span>

# Hold Out
Hold Out is a simple cross validation technique that uses a validation set that is *held out* from the training data. The advantages of Hold Out is that it is quick, but it doesn't allow the model to train and test on the entire training set.

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | ratio | 0.2 | float | The ratio of samples to hold out for testing. |
| 2 | stratify | false | bool | Should we stratify the dataset before splitting? |

### Example
```php
use Rubix\ML\CrossValidation\HoldOut;

$validator = new HoldOut(0.3, true);
```