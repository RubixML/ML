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
     * Return the probabilities of a single sample.
     *
     * @param array $sample
     * @return array
     */
    public function probaSample(array $sample) : array
    {
        $probabilities = $this->proba(Unlabeled::build([$sample]));

        return reset($probabilities);
    }
}
