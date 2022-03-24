<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/CrossValidation/Metrics/ProbabilisticMetric.php">[source]</a></span>

# Probabilistic Distance
This metric comes from the sports betting domain, where it's used to measure the accuracy of predictions by determining the "distance" with a simple formula:
```
Distance = 1 – The Probability for the outcome if it comes up
```
The shorter the distance, the better the accuracy. If you get a forecast completely wrong, you end up with a maximum score of 1. A score of 0 means complete 100% certainty. Let’s say you estimate a probability of a forecast of 80%. If this outcome occurs, the distance is 1 – 0.8 = 0.2.

Based on the <a href="https://mercurius.io/en/learn/predicting-forecasting-football">blog post</a> by Mercurius.

!!! note
    In order to maintain the convention of *maximizing* validation scores, this metric outputs the negative of the original score.

**Output Range:** -1 to 0

## Example
```php
use Rubix\ML\CrossValidation\Metrics\ProbabilisticDistance;

$metric = new ProbabilisticDistance();
```
