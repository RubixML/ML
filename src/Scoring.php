<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Scoring extends Estimator
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
}
