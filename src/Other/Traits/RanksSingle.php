<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Datasets\Unlabeled;

/**
 * Ranks Single
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait RanksSingle
{
    /**
     * Return the score given to a single sample.
     *
     * @param (string|int|float)[] $sample
     * @return float
     */
    public function rankSample(array $sample) : float
    {
        return current($this->rank(Unlabeled::build([$sample]))) ?: NAN;
    }
}
