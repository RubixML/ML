<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Ranking extends Estimator
{
    /**
     * Apply an arbitrary unnormalized scoring function over the dataset.
     *
     * @param \Rubix\ML\Datasets\Dataset<array> $dataset
     * @return float[]
     */
    public function rank(Dataset $dataset) : array;

    /**
     * Return the score given to a single sample.
     *
     * @param mixed[] $sample
     * @return float
     */
    public function rankSample(array $sample) : float;
}
