<?php

namespace Rubix\Engine;

use Rubix\Engine\Datasets\Supervised;

interface Estimator
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;
    const EPSILON = 1e-8;

    /**
     * Train the estimator.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return void
     */
    public function train(Supervised $dataset) : void;

    /**
     * Make a prediction of a given sample.
     *
     * @param  array  $sample
     * @return \Rubix\Engine\Prediction
     */
    public function predict(array $sample) : Prediction;
}
