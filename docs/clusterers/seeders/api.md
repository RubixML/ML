# Seeders
Seeders are responsible for initializing the starting clusters used by certain clusterers such as [K Means](../k-means.md), [Mean Shift](../mean-shift.md), and [Gaussian Mixture](../gaussian-mixture.md). The choice of initializer can play an important role in determining the quality of the final solution derived by a learning algorithm.

### Create Seeds
To create **k** seeds from a dataset:
```php
public function seed(Dataset $dataset, int $k) : array;
```

**Example**

```php
use Rubix\ML\Clusterers\Seeders\PlusPlus;

$seeder = new PlusPlus();

$seeds = $seeder->seed($dataset, 10);
```