<?php

namespace Rubix\ML\Transformers;

use Rubix\ML\Datasets\Dataset;
use InvalidArgumentException;

/**
 * L1 Regularizer
 *
 * Transform each sample vector in the sample matrix such that each feature is
 * scaled by the L1 norm (or magnitude) of that vector.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class L1Regularizer implements Transformer
{
    /**
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(Dataset::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works on'
                . ' continuous features.');
        }
    }

    /**
     * Regularize the dataset by dividing each feature by the L1 norm of the sample
     * vector.
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
                $feature = $norm !== 0.? $feature / $norm : 1.;
            }
        }
    }
}
