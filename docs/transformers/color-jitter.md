# Color Jitter

Adds random jitter to the brightness, contrast, saturation, and hue of images. Color Jitter is commonly used as a data augmentation technique for computer vision tasks. By randomly perturbing the color properties of training images, it helps models become more robust to variations in lighting, exposure, and color balance.

## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| brightness | float | 0.2 | The maximum brightness adjustment factor. When > 0, a brightness factor is sampled uniformly from [1 - brightness, 1 + brightness] and clamped to >= 0. |
| contrast | float | 0.2 | The maximum contrast adjustment factor. When > 0, a contrast factor is sampled uniformly from [1 - contrast, 1 + contrast] and applied to all channels around the image-wide mean luminance (Rec. 601). |
| saturation | float | 0.2 | The maximum saturation adjustment factor. When > 0, a saturation factor is sampled uniformly from [1 - saturation, 1 + saturation]. |
| hue | float | 0.0 | The maximum hue shift in degrees. When > 0, a hue shift is sampled uniformly from [-hue, +hue] degrees. |

## Data Types

Compatible with all data types. Only image-typed values are jittered; non-image columns are left unchanged.

## Example

```php
use Rubix\ML\Transformers\ColorJitter;
use Rubix\ML\Datasets\Unlabeled;

$dataset = new Unlabeled([
    [$image1, 'feature1'],
    [$image2, 'feature2'],
]);

$transformer = new ColorJitter(0.2, 0., 0.2, 30.0);

$dataset->apply($transformer);
```

## Notes

- The order of operations is: brightness → contrast → saturation → hue.
- Random factors are sampled once per image (not per pixel).
- Contrast scales all channels around the image-wide mean luminance (Rec. 601 weights), computed once per image from the source pixels before the pixel loop. Grayscale pixels therefore respond to contrast whenever their luminance differs from the image mean.
- Alpha (transparency) channels are preserved.
