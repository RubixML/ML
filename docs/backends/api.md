# Backend
Backends are used by objects that implement the [Parallel](../parallel.md) interface to carry out their deferred computations. They are usually used to execute batches of computation in parallel as a way of increasing performance.

### Enqueue Computation
To enqueue a Deferred computation for backend processing:
```php
public enqueue(Deferred $deferred, ?Closure $after = null) : void
```

**Example**

```php
use Rubix\ML\Deferred;

$deferred = new Deferred(function ($input) {
    return $input ** 2;
}, 2.5);

$after = function ($result) {
    echo 'done';
};

$backend->enqueue($deferred, $after);
```

### Process Queue
Process the queue of deferred computations:
```php
public process() : array
```

***Example**

```php
$results = $backend->process();

var_dump($results);
```

```sh
array(1) {
    [0]=> float(6.25)
}
```

### Flush Queue
Sometimes it might be necessary to remove leftover items from the queue before proceeding. In such a case the `flush()` method will clear the queue of Deferred computations.
```php
public flush(): void
```

**Example**

```php
$backend->flush();
```