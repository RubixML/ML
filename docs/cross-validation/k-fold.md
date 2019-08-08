<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/KFold.php">Source</a></span>

# K Fold
K Fold is a cross validation technique that splits the training set into k individual folds and for each training round uses 1 of the folds to measure the validation performance of the model and the rest as training data. The final score is then averaged.

**Interfaces:** [Parallel](#parallel)

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 10 | int | The number of times to split the training set into equal sized folds. |
| 2 | stratify | false | bool | Should we stratify the dataset before folding? |

### Example
```php
use Rubix\ML\CrossValidation\KFold;

$validator = new KFold(5, true);
```