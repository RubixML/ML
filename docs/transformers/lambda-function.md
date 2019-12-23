<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/LambdaFunction.php">[source]</a></span>

# Lambda Function
Run a stateless lambda function (*anonymous* function) over the samples. The lambda function receives the sample matrix as an argument and should return the transformed matrix.

**Interfaces:** [Transformer](api.md#transformer)

**Compatibility** Depends on callback function

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | lambda | | callable | The lambda function to run over the sample matrix. |

## Additional Methods
This transformer does not have any additional methods.

## Example
```php
use Rubix\ML\Transformers\LambdaFunction;

$transformer = new LambdaFunction(function ($samples) {
	return array_map(function ($sample) {
		$total = array_sum($sample);
		$mean = $total / count($sample);

		return [$total, $mean];
	}, $samples);
});
```