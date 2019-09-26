# Strategies
Guesses can be thought of as a type of *weak* prediction. Unlike a real prediction, guesses are made using limited information. A guessing Strategy attempts to use such information to formulate an educated guess. Guessing is utilized in both Dummy Estimators ([Dummy Classifier](../../classifiers/dummy-classifier.md), [Dummy Regressor](../../regressors/dummy-regressor.md)) as well as the [Missing Data Imputer](../../transformers/missing-data-imputer.md).

### Fit a Strategy
To fit a Strategy to an array of values:
```php
public fit(array $values) : void
```

**Example**

```php
$strategy->fit($values);
```

### Make a Guess
To make a guess based on the fitted data:
```php
public guess() : mixed
```

**Example**

```php
var_dump($strategy->guess());
```

```sh
string(3) "cat"
```