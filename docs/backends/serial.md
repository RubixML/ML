<p><span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Backends/Serial.php">Source</a></span></p>

### Serial
The Serial backend executes tasks sequentially inside of a single PHP process. The advantage of the Serial backend is that it has zero overhead, thus it may be faster than a parallel backend in cases where the computions are minimal such as with small datasets.

### Parameters
This backend does not have any additional parameters.

### Additional Methods
This backend does not have any additional methods.

### Example
```php
use Rubix\ML\Backends\Serial;

$backend = new Serial();
```