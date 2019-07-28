<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Ranking extends Estimator
{
    /**
     * Apply an arbitrary unnormalized scoring function over the dataset
     * such that the rows can be sorted according to the value.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     */
    public function rank(Dataset $dataset) : array;
}
