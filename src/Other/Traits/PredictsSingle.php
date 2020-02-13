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
     * @param (string|int|float)[] $sample
     * @return mixed
     */
    public function predictSample(array $sample)
    {
        $predictions = $this->predict(Unlabeled::build([$sample]));

        return current($predictions) ?: null;
    }
}
