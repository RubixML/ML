<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Ranking extends Estimator
{
    /**
     * Apply an arbitrary scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function rank(Dataset $dataset) : array;
}
