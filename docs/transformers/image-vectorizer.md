<span style="float:right;"><a href="https://github.com/RubixML/RubixML/blob/master/src/Transformers/ImageVectorizer.php">[source]</a></span>

# Image Vectorizer
Image Vectorizer takes images of the same size and converts them into flat feature vectors of raw color channel intensities. Intensities range from 0 to 255 and can either be read from 1 channel (grayscale) or 3 channels (RGB color) per pixel.

> **Note:** Note that the [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Image

## Parameters
| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | grayscale | false | bool | Should we encode the image in grayscale instead of color? |

## Example
```php
use Rubix\ML\Transformers\ImageVectorizer;

$transformer = new ImageVectorizer(false);
```

## Additional Methods
This transformer does not have any additional methods.
