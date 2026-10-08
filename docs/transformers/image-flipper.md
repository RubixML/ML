<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/ImageFlipper.php">[source]</a></span>

# Image Flipper

Image Flipper permutes an image feature by flipping it along the horizontal and/or vertical axis with a given probability. Since only the position of the pixels changes, the image dimensions are always preserved. Such permutations are useful for training computer vision models that are robust to mirrored data such as digits, faces, or natural scenes.

!!! note
    The [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

!!! note
    The image is flipped in place. Because PHP images are object handles, any other sample or Dataset that refers to the same image will be affected as well.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Image

## Parameters

| # | Name | Default | Type | Description |
| --- | --- | --- | --- | --- |
| 1 | horizontal | 0.5 | float | The probability of flipping the image along the horizontal axis. |
| 2 | vertical | 0.0 | float | The probability of flipping the image along the vertical axis. |

## Example

```php
use Rubix\ML\Transformers\ImageFlipper;

$transformer = new ImageFlipper(0.5, 0.0); // Mirror half of the images left to right.

$transformer = new ImageFlipper(0.5, 0.5); // Mirror half of the images along both axes.
```

## Additional Methods

This transformer does not have any additional methods.