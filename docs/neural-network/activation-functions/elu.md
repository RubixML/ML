### ELU
*Exponential Linear Units* are a type of rectifier that soften the transition from non-activated to activated using the exponential function.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/NeuralNet/ActivationFunctions/ELU.php)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | alpha | 1.0 | float | The value at which leakage will begin to saturate. Ex. alpha = 1.0 means that the output will never be less than -1.0 when inactivated. |

**Example:**

```php
use Rubix\ML\NeuralNet\ActivationFunctions\ELU;

$activationFunction = new ELU(5.0);
```

**References:**

>- D. A. Clevert et al. (2016). Fast and Accurate Deep Network Learning by Exponential Linear Units.