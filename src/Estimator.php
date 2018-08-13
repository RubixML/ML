<?php

namespace Rubix\ML;

use Rubix\ML\Datasets\Dataset;

interface Estimator
{
    const CLASSIFIER = 1;
    const REGRESSOR = 2;
    const CLUSTERER = 3;
    const DETECTOR = 4;

    const EPSILON = 1e-8;

    /**
     * Return the integer encoded type of estimator this is.
     *
     * @return int
     */
    public function type() : int;

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
     * @param  \Rubix\ML\Datasets\Dataset  $dataset
     * @return array
     */
    public function predict(Dataset $dataset) : array;
}
