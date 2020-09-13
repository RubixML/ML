<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Datasets\Unlabeled;

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
        trigger_error('Deprecated, use scoreSample() instead.', E_USER_DEPRECATED);

        return $this->scoreSample($sample);
    }
}
