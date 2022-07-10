<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/ImageResizer.php">[source]</a></span>

# Image Random Rotation
Image rotationer, does a random rotation with minimum of 0 degrees to a maximum specified in the constructor.
The extra image is cropped to fit the original width/height keeping the original center of the image.

!!! note
    The [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

**Interfaces:** [Transformer](api.md#transformer)

**Data Type Compatibility:** Image

## Parameters
| # | Name    | Default | Type | Description                      |
|---|---------|---------|---|----------------------------------|
| 1 | degrees | 30      | int | The maximum degrees of rotation. |

## Example
```php
use Rubix\ML\Transformers\ImageRandomRotationer;

$transformer = new ImageRandomRotationer(45);
```

## Additional Methods
This transformer does not have any additional methods.
