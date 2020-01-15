<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/ImageResizer.php">[source]</a></span>

# Image Resizer
Image Resizer fits (scales and crops) images in a dataset to a user-specified width and height.

> **Note:** The [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Image

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | width | 32 | int | The width of the transformed image. |
| 2 | heights | 32 | int | The height of the transformed image. |

## Additional Methods
This transformer does not have any additional methods.

## Example
```php
use Rubix\ML\Transformers\ImageResizer;

$transformer = new ImageResizer(28, 28, 'gd');
```