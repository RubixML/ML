<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Other/Strategies/PopularityContest.php">Source</a></span>

# Popularity Contest
Hold a popularity contest where the probability of winning (being guessed) is based on the category's prior probability.

**Data Type:** Categorical

### Parameters
This strategy does not have any parameters.

### Example
```php
use Rubix\ML\Other\Strategies\Lottery;

$strategy = new PopularityContest();
```