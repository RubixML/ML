<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

use function is_string;

/**
 * HTML Stripper
 *
 * Removes any HTML tags that may be in the text of a categorical variable.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class HTMLStripper implements Transformer
{
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
                    $value = strip_tags($value);
                }
            }
        }
    }
}
