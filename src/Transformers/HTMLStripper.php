<?php

namespace Rubix\ML\Transformers;

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
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @param  array|null  $labels
     * @throws \RuntimeException
     * @return void
     */
    public function transform(array &$samples, ?array &$labels = null) : void
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
