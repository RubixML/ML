<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Dataset;

interface Estimator
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    const EPSILON = 1e-8;

    /**
     * Make a prediction on a dataset of samples.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array;
}
