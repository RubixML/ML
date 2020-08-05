# Generators
Dataset generators produce synthetic datasets of a user-specified shape and dimensionality. Synthetic data is useful for a number of tasks including experimentation, testing, benchmarking, and demonstration purposes.

### Generate a Dataset
To generate a Dataset object with *n* records:
```php
public generate(int $n) : Dataset
```

```php
use Rubix\ML\Datasets\Generators\HalfMoon;

$generator = new HalfMoon();

$dataset = $generator->generate(1000);
```

### Dimensionality
Return the dimensionality of the samples produced by the generator:
```php
public dimensions() : int
```

```php
var_dump($generator->dimensions());
```

```sh
int(2)
```