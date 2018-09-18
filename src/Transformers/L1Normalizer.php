<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Other\Structures\DataFrame;
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
                $norm += abs($feature);
            }

            foreach ($sample as &$feature) {
                $feature = $norm !== 0. ? $feature / $norm : 1.;
            }
        }
    }
}
