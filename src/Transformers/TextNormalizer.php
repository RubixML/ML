<?php

namespace Rubix\ML\Transformers;

/**
 * Text Normalizer
 *
 * This transformer converts all text to lowercase and removes extra
 * whitespace.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class TextNormalizer implements Transformer
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
                    $feature = preg_replace('/\s+/', ' ', strtolower($feature));
                }
            }
        }
    }
}
