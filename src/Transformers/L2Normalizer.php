<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Other\Structures\DataFrame;
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
     * Fit the transformer to the incoming data frame.
     *
     * @param  \Rubix\ML\Other\Structures\DataFrame  $dataframe
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(DataFrame $dataframe) : void
    {
        if (in_array(DataFrame::CATEGORICAL, $dataframe->types())) {
            throw new InvalidArgumentException('This transformer only works on'
                . ' continuous features.');
        }
    }

    /**
     * Apply the transformation to the samples in the data frame.
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

            $norm = $norm ** 0.5;

            foreach ($sample as &$feature) {
                $feature = $norm !== 0. ? $feature / $norm : 1.;
            }
        }
    }
}
