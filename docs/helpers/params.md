# Params
Generate distributions of values to use in conjunction with [Grid Search](../grid-search.md) or other forms of model selection and/or cross validation.

### Generate Params
To generate a *unique* distribution of integer parameters:
```php
public static ints(int $min, int $max, int $n = 10) : array
```

```php
use Rubix\ML\Helpers\Params;

$ints = Params::ints(0, 100, 5);

print_r($ints);
```

```php
Array
(
    [0] => 88
    [1] => 48
    [2] => 64
    [3] => 100
    [4] => 42
)
```

To generate a random distribution of floating point parameters:
```php
public static floats(float $min, float $max, int $n = 10) : array
```

```php
use Rubix\ML\Helpers\Params;

$floats = Params::floats(0, 100, 5);

print_r($floats);
```

```php
Array
(
    [0] => 42.65728
    [1] => 66.74335
    [2] => 15.17243
    [3] => 71.92631
    [4] => 4.638863
)
```

To generate a uniformly spaced grid of parameters:
```php
public static grid(float $min, float $max, int $n = 10) : array
```

```php
use Rubix\ML\Helpers\Params;

$grid = Params::grid(0, 100, 5);

print_r($grid);
```

```php
Array
(
    [0] => 0
    [1] => 25
    [2] => 50
    [3] => 75
    [4] => 100
)
```
