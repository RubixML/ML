# Params
Generate distributions of values to use in conjunction with [Grid Search](../../grid-search.md) or other forms of model selection and/or cross validation.

### Generate Params
To generate a *unique* distribution of integer parameters:
```php
public static ints(int $min, int $max, int $n = 10) : array
```

```php
use Rubix\ML\Other\Helpers\Params;

$ints = Params::ints(0, 100, 5);

var_dump($ints);
```

```sh
array(5) {
  [0]=> int(88)
  [1]=> int(48)
  [2]=> int(64)
  [3]=> int(100)
  [4]=> int(41)
}
```

To generate a random distribution of floating point parameters:
```php
public static floats(float $min, float $max, int $n = 10) : array
```

```php
use Rubix\ML\Other\Helpers\Params;

$floats = Params::floats(0, 100, 5);

var_dump($floats);
```

```sh
array(5) {
  [0]=> float(42.65728411)
  [1]=> float(66.74335233)
  [2]=> float(15.1724384)
  [3]=> float(71.92631156)
  [4]=> float(4.63886342)
}
```

To generate a uniformly spaced grid of parameters:
```php
public static grid(float $min, float $max, int $n = 10) : array
```

```php
use Rubix\ML\Other\Helpers\Params;

$grid = Params::grid(0, 100, 5);

var_dump($grid);
```

```sh
array(5) {
  [0]=> float(0)
  [1]=> float(25)
  [2]=> float(50)
  [3]=> float(75)
  [4]=> float(100)
}
```