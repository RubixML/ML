### K Fold
K Fold is a technique that splits the training set into K individual sets and for each training round uses 1 of the folds to measure the validation performance of the model. The score is then averaged over K. For example, a K value of 10 will train and test 10 versions of the model using a different testing set each time.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/CrossValidation/KFold.php)

**Interfaces:** [Parallel](#parallel)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | k | 10 | int | The number of times to split the training set into equal sized folds. |
| 2 | stratify | false | bool | Should we stratify the dataset before folding? |

**Example:**

```php
use Rubix\ML\CrossValidation\KFold;

$validator = new KFold(5, true);
```