<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

/**
 * @deprecated
 */
interface Ranking extends Estimator
{
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
