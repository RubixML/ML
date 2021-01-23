<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/KFold.php">[source]</a></span>

# K Fold
K Fold is a cross validation technique that splits the training set into *k* individual folds and for each training round uses 1 of the folds to test the model and the rest as training data. The final score is the average validation score over all of the *k* rounds. K Fold has the advantage of both training and testing on each sample in the dataset at least once.

**Interfaces:** [Validator](api.md#validator), [Parallel](#parallel)

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 5 | int | The number of folds to split the dataset into. |

## Example
```php
use Rubix\ML\CrossValidation\KFold;

$validator = new KFold(5, true);
```