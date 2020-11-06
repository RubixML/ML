<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Datasets\Unlabeled;

use function Rubix\ML\warn_deprecated;

/**
 * Scores Single
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait ScoresSingle
{
    /**
     * Return the anomaly score given to a single sample.
     *
     * @param (string|int|float)[] $sample
     * @return float
     */
    public function scoreSample(array $sample) : float
    {
        return current($this->score(Unlabeled::build([$sample]))) ?: NAN;
    }

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
        warn_deprecated('RankSample() is deprecated, use scoreSample() instead.');

        return $this->scoreSample($sample);
    }
}
