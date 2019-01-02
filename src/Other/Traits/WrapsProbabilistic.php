<?php

namespace Rubix\ML\Other\Traits;

use Rubix\ML\Probabilistic;
use Rubix\ML\Datasets\Dataset;
use RuntimeException;

trait WrapsProbabilistic
{
    /**
     * Estimate probabilities for each possible outcome.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @throws \RuntimeException
     * @return array
     */
    public function proba(Dataset $dataset) : array
    {
        $base = $this->base();

        if ($base instanceof Probabilistic) {
            return $base->proba($dataset);
        }

        throw new RuntimeException('Base estimator must'
            . ' implement the probabilistic interface.');
    }
}