# Params
Generate distributions of values to use in conjunction with [Grid Search](#grid-search) or other forms of model selection and/or cross validation.

To generate a *unique* distribution of integer parameters:
```php
public static ints(int $min, int $max, int $n = 10) : array
```

To generate a random distribution of floating point parameters:
```php
public static floats(float $min, float $max, int $n = 10) : array
```

To generate a uniformly spaced grid of parameters:
```php
public static grid(float $min, float $max, int $n = 10) : array
```

### Example
```php
use Rubix\ML\Other\Helpers\Params;

$ints = Params::ints(0, 100, 5);

$floats = Params::floats(0, 100, 5);

$grid = Params::grid(0, 100, 5);

var_dump($ints);
var_dump($floats);
var_dump($grid);
```

**Output:**

```sh
array(5) {
  [0]=> int(88)
  [1]=> int(48)
  [2]=> int(64)
  [3]=> int(100)
  [4]=> int(41)
}

array(5) {
  [0]=> float(42.65728411)
  [1]=> float(66.74335233)
  [2]=> float(15.1724384)
  [3]=> float(71.92631156)
  [4]=> float(4.63886342)
}

array(5) {
  [0]=> float(0)
  [1]=> float(25)
  [2]=> float(50)
  [3]=> float(75)
  [4]=> float(100)
}

```

### Example
```php
use Rubix\ML\GridSearch;
use Rubix\ML\Other\Helpers\Params;
use Rubix\ML\Clusterers\FuzzyCMeans;
use Rubix\ML\Kernels\Distance\Diagonal;
use Rubix\ML\Kernels\Distance\Minkowski;
use Rubix\CrossValidation\KFold;
use Rubix\CrossValidation\Metrics\VMeasure;

$params = [
	Params::grid(1, 5, 5), Params::floats(1.0, 20.0, 20), [new Diagonal(), new Minkowski(3.0)],
];

$estimator = new GridSearch(FuzzyCMeans::class, $params, new VMeasure(), new KFold(10));

$estimator->train($dataset);

var_dump($estimator->best());
```

**Output:**

```sh
array(3) {
  [0]=> int(4)
  [1]=> float(13.65)
  [2]=> object(Rubix\ML\Kernels\Distance\Diagonal)#15 (0) {
  }
}
```