<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Datasets\Unlabeled;

/**
 * Rank Single
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait RankSingle
{
    /**
     * Return the score given to a single sample.
     *
     * @param (string|int|float)[] $sample
     * @return float
     */
    public function rankSample(array $sample) : float
    {
        $scores = $this->rank(Unlabeled::build([$sample]));

        return reset($scores) ?: NAN;
    }
}
