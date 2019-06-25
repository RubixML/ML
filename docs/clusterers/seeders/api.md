# Seeders
Seeders are responsible for initializing the starting clusters used by certain learners such as [K Means](../k-means.md), [Mean Shift](../mean-shift.md), and [Gaussian Mixture](../guassian-mixture.md).

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