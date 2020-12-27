<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Datasets\Unlabeled;

/**
 * Predicts Single
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait PredictsSingle
{
    /**
     * Predict a single sample and return the result.
     *
     * @internal
     *
     * @param (string|int|float)[] $sample
     * @return mixed
     */
    public function predictSample(array $sample)
    {
        return current($this->predict(Unlabeled::build([$sample]))) ?: null;
    }
}
