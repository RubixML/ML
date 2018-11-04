<?php

namespace Rubix\ML\Transformers;

/**
 * Numeric String Converter
 *
 * This handy Transformer will convert all numeric strings into their floating
 * point counterparts. Useful for when extracting from a source that only
 * recognizes data as string types.
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
     * @param  array  $samples
     * @param  array|null  $labels
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void
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

        if ($labels) {
            foreach ($labels as &$label) {
                if (is_string($label) and is_numeric($label)) {
                    $label = (int) $label == $label
                        ? (int) $label
                        : (float) $label;
                }
            }
        }
    }
}
