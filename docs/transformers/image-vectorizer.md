### Image Vectorizer
Image Vectorizer takes images and converts them into a flat vector of raw color channel data.

> **Note**: Note that the [GD extension](https://php.net/manual/en/book.image.php) is required to use this transformer.

> [Source](https://github.com/RubixML/RubixML/blob/master/src/Transformers/ImageVectorizer.php)

**Interfaces:** [Transformer](#transformers)

**Compatibility:** Resource (Images)

**Parameters:**

| # | Param | Default | Type | Description |
|---|---|---|---|---|
| 1 | channels | 3 | int | The channel depth i.e the number of rgba channels to encode starting with red. |

**Additional Methods:**

This transformer does not have any additional methods.

**Example:**

```php
use Rubix\ML\Transformers\ImageVectorizer;

$transformer = new ImageVectorizer(3);
```