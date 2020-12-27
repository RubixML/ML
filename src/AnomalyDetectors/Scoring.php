<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Estimator;
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
}
