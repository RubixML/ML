<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

/**
 * Whitespace Trimmer
 *
 * Trims extra whitespace from all strings in the dataset.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class WhitespaceTrimmer implements Transformer
{
    /**
     * A pattern to match whitespace.
     *
     * @var string
     */
    protected const SPACES_REGEX = '/\s+/';

    /**
     * A whitespace character.
     *
     * @var string
     */
    protected const SPACE = ' ';

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
                    $value = preg_replace(self::SPACES_REGEX, self::SPACE, trim($value));
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
        return 'Whitespace Trimmer';
    }
}
