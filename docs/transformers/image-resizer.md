<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/ImageResizer.php">Source</a></span>

# Image Resizer
The Image Resizer scales and crops images to a user specified width, height, and color depth.

> **Note:** Note that the [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Resource (GD Image)

### Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | width | 32 | int | The width of the transformed image. |
| 2 | heights | 32 | int | The height of the transformed image. |
| 3 | driver | 'gd' | string | The PHP extension to use for image processing ('gd' *or* 'imagick'). |

### Additional Methods
This transformer does not have any additional methods.

### Example
```php
use Rubix\ML\Transformers\ImageResizer;

$transformer = new ImageResizer(28, 28, 'gd');
```