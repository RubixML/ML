<?php

namespace Rubix\Engine\Preprocessors;

use Rubix\Engine\Dataset;

class L2Normalizer implements Preprocessor
{
    const EPSILON = 1e-10;

    /**
     * @param  \Rubix\Engine\Dataset  $data
     * @return void
     */
    public function fit(Dataset $data) : void
    {
        //
    }

    /**
     * Normalize the dataset.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $norm = sqrt(array_reduce($sample, function ($carry, $feature) {
                return $carry += $feature ** 2;
            }, 0)) + self::EPSILON;

            foreach ($sample as &$feature) {
                $feature /= $norm;
            }
        }
    }
}
