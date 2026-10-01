<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/ImageRotator.php">[source]</a></span>

# Image Rotator

Image Rotator permutes an image feature by rotating it by a given offset angle and adding optional randomized jitter. The rotated image is then resized back to the original width and height, maintaining the dimensionality. Permutations such as these are useful for training computer vision models that are robust to rotation and small variations in orientation.

!!! note
    The [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Image

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | offset | 0.0 | float | The angle of the rotation in degrees. |
| 2 | jitter | 0.2 | float | The proportion of a half-turn (180 degrees) of random jitter to apply to the rotation in either direction. |
| 3 | fillColor | #000000 | ?string | The hex color used to fill the area exposed by rotation, such as `'#ffffff'`. Null uses black. |

## Example

```php
use Rubix\ML\Transformers\ImageRotator;

$transformer = new ImageRotator(-90.0); // Rotate 90 degrees clockwise.

$transformer = new ImageRotator(0.0, 0.5); // Add random jitter about the origin.

$transformer = new ImageRotator(0.0, 0.2, '#ffffff'); // Fill exposed area with white.
```

!!! note
    GD reads the background argument to `imagerotate()` as a palette index for palette images
    (such as GIF and PNG-8) and as an RGB value for truecolor images. This transformer resolves
    the color correctly for both, so the same hex value works regardless of image type.

## Additional Methods

This transformer does not have any additional methods.
