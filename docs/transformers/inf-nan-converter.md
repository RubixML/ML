<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/InfNanConverter.php">[source]</a></span>

# INF/NAN converter
This transformer is used to convert `NAN` and `INF` constants (both positive and negative) to their arbitrary string 
equivalent and back. The main goal of the converter is transforming dataset samples before sending them as JSON request
(`json_encode` does not support these constants). The transformer is reversible, so after receiving the samples you 
can decode the constants back.

**Interfaces:** [Transformer](api.md#transformer), [Reversible](api.md#reversible)

**Data Type Compatibility:** Categorical, Continuous

## Example
Client side:
```php
use Rubix\ML\Transformers\InfNanConverter;

$dataset->apply(new InfNanConverter());
// now you can encode $dataset->samples() and send as part of a JSON HTTP request.
```

Client side:
```php
use Rubix\ML\Transformers\InfNanConverter;

// build $dataset from decoded JSON payload and reverse apply the converter to get constant values back.
$dataset->reverseApply(new InfNanConverter());
```

## Additional Methods
This transformer does not have any additional methods.
