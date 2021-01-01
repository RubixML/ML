<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Datasets\Unlabeled;

use function Rubix\ML\warn_deprecated;

/**
 * Ranks Single
 *
 * @deprecated
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
     * @deprecated
     *
     * @param (string|int|float)[] $sample
     * @return float
     */
    public function rankSample(array $sample) : float
    {
        warn_deprecated('RankSample() is deprecated and will be removed in the next major release.');

        return current($this->score(Unlabeled::build([$sample]))) ?: NAN;
    }
}
