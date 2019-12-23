# Generators
Dataset generators produce synthetic datasets of a user-specified shape and dimensionality. Synthetic data is useful for a number of tasks including experimentation, testing, benchmarking, and demonstration purposes.

### Generate a Dataset
To generate a Dataset object with *n* rows:
```php
public generate(int $n) : Dataset
```

**Example**

```php
use Rubix\ML\Datasets\Generators\Blob;

$generator = new Blob([0, 0], 1.0);

$dataset = $generator->generate(3);

var_dump($dataset);
```

```sh
object(Rubix\ML\Datasets\Unlabeled) {
  ["samples":protected]=>
  array(3) {
    [0]=>
    array(2) {
      [0]=> float(-0.2729673885539)
      [1]=> float(0.43761840244204)
    }
    [1]=>
    array(2) {
      [0]=> float(-1.2718092282012)
      [1]=> float(-1.9558245484829)
    }
    [2]=>
    array(2) {
      [0]=> float(1.1774185431405)
      [1]=> float(0.05168623824664)
    }
  }
}
```

### Dimensionality
Return the dimensionality of the samples produced by the generator:
```php
public dimensions() : int
```

**Example**

```php
var_dump($generator->dimensions());
```

```sh
int(2)
```