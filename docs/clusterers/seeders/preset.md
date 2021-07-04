<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Clusterers/Seeders/Preset.php">[source]</a></span>

# Preset
Generates centroids from a list of presets.

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | centroids| | array |  A list of predefined cluster centroids to sample from. |

## Example
```php
use Rubix\ML\Clusterers\Seeders\Preset;

$seeder = new Preset([
    ['foo', 14, 0.72],
    ['bar', 16, 0.92],
]);
```
