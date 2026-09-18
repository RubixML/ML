<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/PowerTransformer.php">[source]</a></span>

# Power Transformer

A family of parametric, monotonic transformations that apply the [Yeo-Johnson](https://en.wikipedia.org/wiki/Power_transform) transformation to the features of a dataset to make their distributions more Gaussian-like. The transformation parameter (lambda) is estimated per feature via maximum likelihood. Unlike the Box-Cox family, the Yeo-Johnson transformation is defined for negative and zero values.

$$
{\displaystyle z = \begin{cases} { (x + 1)^\lambda - 1 \over \lambda } & x \ge 0, \lambda \neq 0 \\ \ln{(x + 1)} & x \ge 0, \lambda = 0 \\ -{ ((1 - x)^{2 - \lambda} - 1) \over 2 - \lambda } & x < 0, \lambda \neq 2 \\ -\ln{(1 - x)} & x < 0, \lambda = 2 \end{cases}}
$$

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful), [Reversible](api.md#reversible), [Persistable](../persistable.md)

**Data Type Compatibility:** Continuous

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | lambda | null | float | A fixed transformation parameter to apply to all features. If null, the lambda is estimated per feature via maximum likelihood during fitting. |

## Example

```php
use Rubix\ML\Transformers\PowerTransformer;

$transformer = new PowerTransformer();
```

## Additional Methods

Return the estimated transformation parameters indexed by column.

```php
public lambdas() : array
```

## References

[^1]: I. K. Yeo, R. A. Johnson (2000). A New Family of Power Transformations to Improve Normality or Symmetry.