<span style="float:right;"><a href="https://github.com/RubixML/ML/blob/master/src/Transformers/ImageVectorizer.php">[source]</a></span>

# Image Vectorizer
Image Vectorizer takes images of the same size and converts them into flat feature vectors of raw color channel intensities. Intensities range from 0 to 255 and can either be read from any of the RGB color channels.

!!! note
    Note that the [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

**Interfaces:** [Transformer](api.md#transformer), [Stateful](api.md#stateful)

**Data Type Compatibility:** Image

## Parameters
| # | Name | Default | Type | Description |
|---|---|---|---|---|
| 1 | red | true | bool | Should we encode the red channel? |
| 1 | green | true | bool | Should we encode the green channel? |
| 1 | blue | true | bool | Should we encode the blue channel? |

## Example
```php
use Rubix\ML\Transformers\ImageVectorizer;

$transformer = new ImageVectorizer(true, true, true);
```

## Additional Methods
This transformer does not have any additional methods.
