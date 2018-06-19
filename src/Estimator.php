<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Estimator
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    const EPSILON = 1e-8;

    /**
     * Train the estimator with a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return void
     */
    public function train(Dataset $dataset) : void;

    /**
     * Make a prediction from a dataset.
     *
     * @param  \Rubix\ML\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $dataset) : array;
}
