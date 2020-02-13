<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Datasets\Unlabeled;

/**
 * Proba Single
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait ProbaSingle
{
    /**
     * Predict the probabilities of a single sample and return the joint distribution.
     *
     * @param (string|int|float)[] $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array
    {
        $probabilities = $this->proba(Unlabeled::build([$sample]));

        return current($probabilities) ?: [];
    }
}
