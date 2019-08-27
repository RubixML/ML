<?php

namespace Rubix\ML\Transformers;

/**
 * Numeric String Converter
 *
 * Convert all numeric strings into their integer and floating point
 * countertypes. Useful for when extracting from a source that only recognizes
 * data as string types.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class NumericStringConverter implements Transformer
{
    /**
     * Transform the dataset in place.
     *
     * @param array $samples
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            foreach ($sample as &$feature) {
                if (is_string($feature) and is_numeric($feature)) {
                    $feature = (int) $feature == $feature
                        ? (int) $feature
                        : (float) $feature;
                }
            }
        }
    }
}
