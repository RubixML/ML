<?php

namespace Rubix\ML\Transformers;

use InvalidArgumentException;

/**
 * L1 Normalizer
 *
 * Transform each sample vector in the sample matrix such that each feature is
 * divided by the L1 norm (or magnitude) of that vector.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class L1Normalizer implements Transformer
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
            $norm = array_sum(array_map('abs', $sample)) ?: self::EPSILON;

            foreach ($sample as &$feature) {
                $feature /= $norm;
            }
        }
    }
}
