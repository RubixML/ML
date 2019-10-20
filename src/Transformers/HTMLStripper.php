<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\DataType;

/**
 * HTML Stripper
 *
 * Removes any HTML tags that may be in the text of a categorical
 * variable.
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
     * @return int[]
     */
    public function compatibility() : array
    {
        return DataType::ALL;
    }

    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$feature) {
                if (is_string($feature)) {
                    $feature = strip_tags($feature);
                }
            }
        }
    }
}
