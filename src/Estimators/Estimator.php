<?php

namespace Rubix\Engine\Estimators;

use Rubix\Engine\Datasets\Dataset;
use Rubix\Engine\Datasets\Supervised;
use Rubix\Engine\Estimators\Predictions\Prediction;

interface Estimator
{
    const CATEGORICAL = 1;
    const CONTINUOUS = 2;

    const EPSILON = 1e-8;

    /**
     * Train the estimator with a supervised dataset.
     *
     * @param  \Rubix\Engine\Datasets\Supervised  $dataset
     * @return void
     */
    public function train(Supervised $dataset) : void;

    /**
     * Make a prediction on a dataset of samples.
     *
     * @param  \Rubix\Engine\Datasets\Dataset  $samples
     * @return array
     */
    public function predict(Dataset $samples) : array;
}
