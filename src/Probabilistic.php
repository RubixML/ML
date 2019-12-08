<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Probabilistic extends Estimator
{
    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param \Rubix\ML\Datasets\Dataset $dataset
     * @return array[]
     */
    public function proba(Dataset $dataset) : array;

    /**
     * Return the probabilities of a single sample.
     *
     * @param mixed[] $sample
     * @return float[]
     */
    public function probaSample(array $sample) : array;
}
