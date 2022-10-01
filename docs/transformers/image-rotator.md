<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/ImageRotator.php">[source]</a></span>

# Image Rotator
Image Rotator permutes an image feature by rotating it and adding optional randomized jitter. The image is then cropped to fit the original width and height maintaining the dimensionality. Permutations such as these are useful for training computer vision models that are robust to 

!!! note
    The [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Image

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | offset | | float | The angle of the rotation in degrees. |
| 2 | jitter | 0.0 | float | The amount of random jitter to apply to the rotation. |

## Example
```php
use Rubix\ML\Transformers\ImageRotator;

$transformer = new ImageRotator(-90.0); // Rotate 90 degrees clockwise.

$transformer = new ImageRotator(0.0, 0.5); // Add random jitter about the origin.
```

## Additional Methods
This transformer does not have any additional methods.
