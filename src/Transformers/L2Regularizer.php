<?php

namespace Rubix\Engine\Transformers;

use Rubix\Engine\Datasets\Dataset;
use InvalidArgumentException;

class L2Regularizer implements Transformer
{
    /**
     * @param  \Rubix\Engine\Datasets\Dataset  $dataset
     * @throws \InvalidArgumentException
     * @return void
     */
    public function fit(Dataset $dataset) : void
    {
        if (in_array(self::CATEGORICAL, $dataset->columnTypes())) {
            throw new InvalidArgumentException('This transformer only works on'
                . ' continuous features.');
        }
    }

    /**
     * Regularize the dataset by dividing each feature by the L2 norm of the sample
     * vector.
     *
     * @param  array  $samples
     * @return void
     */
    public function transform(array &$samples) : void
    {
        foreach ($samples as &$sample) {
            $norm = 0.0;

            foreach ($sample as &$feature) {
                $norm += $feature ** 2;
            }

            $norm = sqrt($norm);

            foreach ($sample as &$feature) {
                $feature /= ($norm + self::EPSILON);
            }
        }
    }
}
