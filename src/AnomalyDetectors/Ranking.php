<?php

namespace Rubix\ML\AnomalyDetectors;

use Rubix\ML\Estimator;
use Rubix\ML\Datasets\Dataset;

interface Ranking extends Estimator
{
    /**
     * Apply an arbitrary unnormalized scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function rank(Dataset $dataset) : array;
}
