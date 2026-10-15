<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/TargetEncoder.php">[source]</a></span>

# Target Encoder

The Target Encoder converts categorical feature columns into continuous representations using statistics derived from the label, or *target*, of a supervised dataset. For a continuous target each category is replaced by the Bayesian-smoothed mean of the target over that category, leaving the column count unchanged. For a categorical target each category is replaced by the Bayesian-smoothed probability of each class, expanding a single column into one column per class. Smoothing pulls each per-category estimate toward the global estimate, which guards against rare categories dominating the fit. Categories not observed during fitting are encoded with the global prior.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Persistable](../persistable.md)

**Data Type Compatibility:** Categorical, Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | smoothing | 1.0 | float | The amount of Bayesian smoothing applied to each per-category estimate. Higher values shrink rare categories toward the global prior, reducing variance at the cost of bias, while lower values fit categories with few samples more aggressively. |

## Example

```php
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Transformers\TargetEncoder;

$dataset = Labeled::build(
    samples: [
        ['red', 'small'],
        ['red', 'large'],
        ['blue', 'small'],
    ],
    labels: [0.1, 0.9, 0.4]
);

$transformer = new TargetEncoder(smoothing: 10.0);

$dataset->apply($transformer);
```

## Additional Methods

Return the per-category encodings computed during fitting, indexed by feature column.

```php
public encodings() : ?array
```

Return the global prior used as a fallback for categories not observed during fitting.

```php
public globalPrior() : ?array
```
