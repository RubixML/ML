<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

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
     * @param list<list<mixed>> $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$value) {
                if (is_string($value)) {
                    $value = mb_strtolower($value);
                }
            }
        }
    }

    /**
     * Return the string representation of the object.
     *
     * @return string
     */
    public function __toString() : string
    {
        return 'Multibyte Text Normalizer';
    }
}
