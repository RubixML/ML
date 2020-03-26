<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

/**
 * Text Normalizer
 *
 * This transformer converts all text to lowercase and removes extra whitespace.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TextNormalizer implements Transformer
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
     * @return \Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return DataType::all();
    }

    /**
     * Transform the dataset in place.
     *
     * @param array[] $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$value) {
                if (is_string($value)) {
                    $value = strtolower(preg_replace(self::SPACES_REGEX, self::SPACE, trim($value)) ?: '');
                }
            }
        }
    }
}
