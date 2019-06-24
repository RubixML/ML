### Lambda Function
Run a stateless lambda function (*anonymous* function) over the sample matrix. The lambda function receives the sample matrix (and labels if applicable) as an argument and should return the transformed sample matrix and labels in a [2-tuple](#what-is-a-tuple).

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/LambdaFunction.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility** Depends on user function

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | lambda | | callable | The lambda function to run over the sample matrix. |

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\LambdaFunction;

$transformer = new LambdaFunction(function ($samples, $labels) {
	$samples = array_map(function ($sample) {
		return [array_sum($sample)];
	}, $samples);

	return [$samples, $labels];
});
```