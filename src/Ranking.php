<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Ranking extends Estimator
{
    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return float[]
     */
    public function rank(Dataset $dataset) : array;

    /**
     * Return the score given to a single sample.
     *
     * @param (string|int|float)[] $sample
     * @return float
     */
    public function rankSample(array $sample) : float;
}
