<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/LambdaFunction.php">[source]</a></span>

# Lambda Function
 Run a stateless lambda function over the samples in a dataset. The function receives three arguments - the sample to be transformed, its row offset in the dataset, and a user-defined outside context variable that can be used to hold state.

**Note:** If the transformation results in a change in dimensionality, the change must be consistent for each sample.

**Interfaces:** [Transformer](api.md#transformer)

**Compatibility** Depends on callback function

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | callback | | callable | The function to call over the samples in the dataset. |
| 2 | context | null | mixed | The outside context that gets injected into the callback function on each call. |

## Example
```php
use Rubix\ML\Transformers\LambdaFunction;

$callback = function (&$sample, $offset, $context) {
	$sample[] = log1p($sample[3]);
};

$transformer = new LambdaFunction($callback, 'example context');
```

## Additional Methods
This transformer does not have any additional methods.
