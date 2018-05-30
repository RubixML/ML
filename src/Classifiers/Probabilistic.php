<?php

namespace Rubix\Engine\Classifiers;

use Rubix\Engine\Datasets\Dataset;

interface Probabilistic
{
    /**
     * Output a vector of class probabilities per sample.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function proba(Dataset $samples) : array;
}
