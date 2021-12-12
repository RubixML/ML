<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function is_string;
use function array_walk;
use function call_user_func;

/**
 * Multibyte Text Normalizer
 *
 * This transformer converts the characters in all multibyte strings to lowercase. Multibyte
 * strings contain characters such as accents (é, è, à), emojis (😀, 😉) or characters of
 * non roman alphabets such as Chinese and Cyrillic.
 *
 * > **Note:** ⚠️ We recommend you install the mbstring extension for best performance.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Maxime Colin
 */
class MultibyteTextNormalizer implements Transformer
{
    /**
     * The normalization function.
     *
     * @var callable-string
     */
    protected string $normalize;

    /**
     * @param bool $uppercase
     */
    public function __construct(bool $uppercase = false)
    {
        $this->normalize = $uppercase ? 'mb_strtoupper' : 'mb_strtolower';
    }

    /**
     * Return the data types that this transformer is compatible with.
     *
     * @internal
     *
     * @return list<\Rubix\ML\DataType>
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array<mixed[]> $samples
     */
    public function transform(array &$samples) : void
    {
        array_walk($samples, [$this, 'normalize']);
    }

    /**
     * Normalize the text in a sample.
     *
     * @param list<mixed> $sample
     */
    public function normalize(array &$sample) : void
    {
        foreach ($sample as &$value) {
            if (is_string($value)) {
                $value = call_user_func($this->normalize, $value);
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @internal
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Multibyte Text Normalizer';
    }
}
