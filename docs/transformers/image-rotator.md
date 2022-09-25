<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/ImageRotator.php">[source]</a></span>

# Image Rotator
Image Rotator permutes an image feature by adding rotational jitter to the original. The image is then cropped to fit the original width and height maintaining the number of pixels and the original center.

!!! note
    The [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Image

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | offset | 0.0 | float | The angle of the rotation in degrees. |
| 2 | jitter | 1.0 | float | The amount of jitter to apply to the rotation. |

## Example
```php
use Rubix\ML\Transformers\ImageRotator;

$transformer = new ImageRotator(-90.0, 180.0);
```

## Additional Methods
This transformer does not have any additional methods.
