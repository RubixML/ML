<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

/**
 * Ranking
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
interface Ranking extends Estimator
{
    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return float[]
     */
    public function score(Dataset $dataset) : array;

    /**
     * Return the anomaly score given to a single sample.
     *
     * @param (string|int|float)[] $sample
     * @return float
     */
    public function scoreSample(array $sample) : float;

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @deprecated
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return float[]
     */
    public function rank(Dataset $dataset) : array;

    /**
     * Return the score given to a single sample.
     *
     * @deprecated
     *
     * @param (string|int|float)[] $sample
     * @return float
     */
    public function rankSample(array $sample) : float;
}
