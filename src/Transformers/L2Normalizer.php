<?php

namespace Rubix\ML\Transformers;

use InvalidArgumentException;

/**
 * L2 Normalizer
 *
 * Transform each sample vector in the sample matrix such that each feature is
 * divided by the L2 norm (or magnitude) of that vector.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class L2Normalizer implements Transformer
{
    /**
     * Transform the dataset in place.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $norm = 0.;

            foreach ($sample as &$feature) {
                $norm += $feature ** 2;
            }

            $norm = sqrt($norm ?: self::EPSILON);

            foreach ($sample as &$feature) {
                $feature /= $norm;
            }
        }
    }
}
